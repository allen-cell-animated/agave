#include "CacheManager.h"

#include "ImageXYZC.h"
#include "Logging.h"

#include "tensorstore/array.h"
#include "tensorstore/context.h"
#include "tensorstore/open.h"
#include "tensorstore/tensorstore.h"

// must include after tensorstore so that tensorstore picks up its own internal json impl
#include "json/json.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <functional>
#include <fstream>
#include <limits>
#include <sstream>

namespace {

inline void
hashCombine(std::size_t& seed, std::size_t value)
{
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

std::uint64_t
nowMillis()
{
  return static_cast<std::uint64_t>(
    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}

std::string
toHex(std::size_t value)
{
  std::ostringstream stream;
  stream << std::hex << value;
  return stream.str();
}

std::vector<std::string>
channelNamesFromImage(const ImageXYZC& image)
{
  std::vector<std::string> names;
  names.reserve(image.sizeC());
  for (size_t i = 0; i < image.sizeC(); ++i) {
    names.push_back(image.channel(static_cast<uint32_t>(i))->m_name);
  }
  return names;
}

} // namespace

bool
CacheKey::operator==(const CacheKey& other) const
{
  return filepath == other.filepath && subpath == other.subpath && scene == other.scene && time == other.time &&
         channels == other.channels && minx == other.minx && maxx == other.maxx && miny == other.miny &&
         maxy == other.maxy && minz == other.minz && maxz == other.maxz && isImageSequence == other.isImageSequence;
}

std::size_t
CacheKeyHash::operator()(const CacheKey& key) const
{
  std::size_t seed = 0;
  hashCombine(seed, std::hash<std::string>{}(key.filepath));
  hashCombine(seed, std::hash<std::string>{}(key.subpath));
  hashCombine(seed, std::hash<int>{}(key.scene));
  hashCombine(seed, std::hash<std::uint32_t>{}(key.time));
  hashCombine(seed, std::hash<std::uint32_t>{}(key.minx));
  hashCombine(seed, std::hash<std::uint32_t>{}(key.maxx));
  hashCombine(seed, std::hash<std::uint32_t>{}(key.miny));
  hashCombine(seed, std::hash<std::uint32_t>{}(key.maxy));
  hashCombine(seed, std::hash<std::uint32_t>{}(key.minz));
  hashCombine(seed, std::hash<std::uint32_t>{}(key.maxz));
  hashCombine(seed, std::hash<bool>{}(key.isImageSequence));
  for (auto ch : key.channels) {
    hashCombine(seed, std::hash<std::uint32_t>{}(ch));
  }
  return seed;
}

CacheManager&
CacheManager::instance()
{
  static CacheManager manager;
  return manager;
}

void
CacheManager::setConfig(const CacheConfig& config)
{
  CacheConfig configCopy;
  bool rebuildDiskIndex = false;
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_config = config;
    if (!m_config.enabled) {
      m_entries.clear();
      m_lruKeys.clear();
      m_currentRamBytes = 0;
      m_diskEntries.clear();
      m_currentDiskBytes = 0;
      m_diskIndexRoot.clear();
      return;
    }

    if (!m_config.enableDisk || m_config.cacheDir.empty()) {
      m_diskEntries.clear();
      m_currentDiskBytes = 0;
      m_diskIndexRoot.clear();
    } else if (m_diskIndexRoot != m_config.cacheDir) {
      m_diskEntries.clear();
      m_currentDiskBytes = 0;
      m_diskIndexRoot = m_config.cacheDir;
      rebuildDiskIndex = true;
      configCopy = m_config;
    }
  }

  if (rebuildDiskIndex) {
    loadDiskIndex(configCopy);
    evictDiskIfNeeded(configCopy, 0);
  }

  evictIfNeeded(0);
}

CacheConfig
CacheManager::getConfig() const
{
  std::lock_guard<std::mutex> lock(m_mutex);
  return m_config;
}

std::shared_ptr<ImageXYZC>
CacheManager::findImage(const LoadSpec& loadSpec)
{
  CacheConfig configCopy;
  CacheKey key = makeKey(loadSpec);
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    configCopy = m_config;
    if (m_config.enabled && m_config.maxRamBytes > 0) {
      auto it = m_entries.find(key);
      if (it != m_entries.end()) {
        touchEntry(it->second.lruIt);
        m_stats.ramHits++;
        LOG_DEBUG << "Cache stats: ram_hits=" << m_stats.ramHits << " disk_hits=" << m_stats.diskHits
                  << " misses=" << m_stats.misses << " disk_writes=" << m_stats.diskWrites;
        return it->second.image;
      }
    }
  }

  if (configCopy.enabled && configCopy.enableDisk && configCopy.maxDiskBytes > 0) {
    auto diskImage = loadFromDisk(key, configCopy);
    if (diskImage) {
      {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stats.diskHits++;
        LOG_DEBUG << "Cache stats: ram_hits=" << m_stats.ramHits << " disk_hits=" << m_stats.diskHits
                  << " misses=" << m_stats.misses << " disk_writes=" << m_stats.diskWrites;
      }
      storeImageInMemory(key, diskImage);
      return diskImage;
    }
  }

  {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_stats.misses++;
    LOG_DEBUG << "Cache stats: ram_hits=" << m_stats.ramHits << " disk_hits=" << m_stats.diskHits
              << " misses=" << m_stats.misses << " disk_writes=" << m_stats.diskWrites;
  }

  return nullptr;
}

void
CacheManager::storeImage(const LoadSpec& loadSpec, const std::shared_ptr<ImageXYZC>& image)
{
  if (!image) {
    return;
  }

  CacheConfig configCopy;
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    configCopy = m_config;
  }

  if (configCopy.enabled && configCopy.enableDisk && configCopy.maxDiskBytes > 0) {
    storeToDisk(makeKey(loadSpec), image, configCopy);
  }

  storeImageInMemory(makeKey(loadSpec), image);
}

void
CacheManager::clear()
{
  std::lock_guard<std::mutex> lock(m_mutex);
  m_entries.clear();
  m_lruKeys.clear();
  m_currentRamBytes = 0;
}

void
CacheManager::clearDiskCache()
{
  CacheConfig configCopy;
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    configCopy = m_config;
    m_diskEntries.clear();
    m_currentDiskBytes = 0;
  }

  if (configCopy.cacheDir.empty()) {
    return;
  }

  std::error_code ec;
  std::filesystem::remove_all(configCopy.cacheDir, ec);
}

CacheManager::CacheStats
CacheManager::getStats() const
{
  std::lock_guard<std::mutex> lock(m_mutex);
  return m_stats;
}

void
CacheManager::resetStats()
{
  std::lock_guard<std::mutex> lock(m_mutex);
  m_stats = CacheStats{};
}

CacheKey
CacheManager::makeKey(const LoadSpec& loadSpec) const
{
  CacheKey key;
  key.filepath = loadSpec.filepath;
  key.subpath = loadSpec.subpath;
  key.scene = loadSpec.scene;
  key.time = loadSpec.time;
  key.channels = loadSpec.channels;
  key.minx = loadSpec.minx;
  key.maxx = loadSpec.maxx;
  key.miny = loadSpec.miny;
  key.maxy = loadSpec.maxy;
  key.minz = loadSpec.minz;
  key.maxz = loadSpec.maxz;
  key.isImageSequence = loadSpec.isImageSequence;
  return key;
}

std::string
CacheManager::keyToString(const CacheKey& key) const
{
  std::ostringstream stream;
  stream << key.filepath << "|" << key.subpath << "|" << key.scene << "|" << key.time << "|";
  stream << key.minx << "," << key.maxx << "," << key.miny << "," << key.maxy << "," << key.minz << "," << key.maxz
         << "|" << (key.isImageSequence ? 1 : 0) << "|";
  for (size_t i = 0; i < key.channels.size(); ++i) {
    if (i > 0) {
      stream << ",";
    }
    stream << key.channels[i];
  }
  return stream.str();
}

std::string
CacheManager::diskCacheId(const CacheKey& key) const
{
  std::size_t hashValue = std::hash<std::string>{}(keyToString(key));
  return toHex(hashValue);
}

std::uint64_t
CacheManager::estimateImageBytes(const ImageXYZC& image) const
{
  std::uint64_t bytesPerPixel = static_cast<std::uint64_t>(ImageXYZC::IN_MEMORY_BPP / 8);
  std::uint64_t numPixels = static_cast<std::uint64_t>(image.sizeX()) * static_cast<std::uint64_t>(image.sizeY()) *
                            static_cast<std::uint64_t>(image.sizeZ()) * static_cast<std::uint64_t>(image.sizeC());
  return numPixels * bytesPerPixel;
}

void
CacheManager::touchEntry(std::list<CacheKey>::iterator it)
{
  if (it == m_lruKeys.begin()) {
    return;
  }
  m_lruKeys.splice(m_lruKeys.begin(), m_lruKeys, it);
}

void
CacheManager::evictIfNeeded(std::uint64_t incomingBytes)
{
  if (m_config.maxRamBytes == 0) {
    return;
  }

  while (!m_lruKeys.empty() && (m_currentRamBytes + incomingBytes) > m_config.maxRamBytes) {
    const CacheKey& key = m_lruKeys.back();
    auto it = m_entries.find(key);
    if (it != m_entries.end()) {
      m_currentRamBytes -= it->second.bytes;
      m_entries.erase(it);
    }
    m_lruKeys.pop_back();
  }
}

void
CacheManager::storeImageInMemory(const CacheKey& key, const std::shared_ptr<ImageXYZC>& image)
{
  std::lock_guard<std::mutex> lock(m_mutex);
  if (!m_config.enabled || m_config.maxRamBytes == 0) {
    return;
  }

  std::uint64_t bytes = estimateImageBytes(*image);
  if (bytes == 0 || bytes > m_config.maxRamBytes) {
    return;
  }

  auto existing = m_entries.find(key);
  if (existing != m_entries.end()) {
    m_currentRamBytes -= existing->second.bytes;
    m_lruKeys.erase(existing->second.lruIt);
    m_entries.erase(existing);
  }

  evictIfNeeded(bytes);

  m_lruKeys.push_front(key);
  CacheEntry entry;
  entry.image = image;
  entry.bytes = bytes;
  entry.lruIt = m_lruKeys.begin();
  m_entries.emplace(key, entry);
  m_currentRamBytes += bytes;
}

std::shared_ptr<ImageXYZC>
CacheManager::loadFromDisk(const CacheKey& key, const CacheConfig& config)
{
  if (!config.enableDisk || config.cacheDir.empty()) {
    return nullptr;
  }

  std::filesystem::path entryPath = std::filesystem::path(config.cacheDir) / diskCacheId(key);
  std::filesystem::path metaPath = entryPath / "meta.json";
  std::filesystem::path dataPath = entryPath / "data.zarr";
  if (!std::filesystem::exists(metaPath) || !std::filesystem::exists(dataPath)) {
    return nullptr;
  }

  nlohmann::json meta;
  try {
    std::ifstream metaFile(metaPath.string());
    metaFile >> meta;
  } catch (...) {
    return nullptr;
  }

  if (!meta.contains("key") || meta["key"].get<std::string>() != keyToString(key)) {
    return nullptr;
  }

  if (!meta.contains("sizeX") || !meta.contains("sizeY") || !meta.contains("sizeZ") || !meta.contains("sizeC")) {
    return nullptr;
  }

  std::uint32_t sizeX = meta["sizeX"].get<std::uint32_t>();
  std::uint32_t sizeY = meta["sizeY"].get<std::uint32_t>();
  std::uint32_t sizeZ = meta["sizeZ"].get<std::uint32_t>();
  std::uint32_t sizeC = meta["sizeC"].get<std::uint32_t>();
  std::uint32_t bpp = meta.value("bpp", static_cast<std::uint32_t>(ImageXYZC::IN_MEMORY_BPP));
  float sx = meta.value("physicalSizeX", 1.0f);
  float sy = meta.value("physicalSizeY", 1.0f);
  float sz = meta.value("physicalSizeZ", 1.0f);
  std::string spatialUnits = meta.value("spatialUnits", std::string("units"));

  std::uint64_t bytes = static_cast<std::uint64_t>(sizeX) * static_cast<std::uint64_t>(sizeY) *
                        static_cast<std::uint64_t>(sizeZ) * static_cast<std::uint64_t>(sizeC) *
                        static_cast<std::uint64_t>(bpp / 8);
  if (bytes == 0) {
    return nullptr;
  }

  std::unique_ptr<uint8_t[]> data(new uint8_t[bytes]);

  auto openFuture =
    tensorstore::Open({ { "driver", "zarr3" }, { "kvstore", { { "driver", "file" }, { "path", dataPath.string() } } } },
                      tensorstore::OpenMode::open,
                      tensorstore::ReadWriteMode::read);
  auto result = openFuture.result();
  if (!result.ok()) {
    return nullptr;
  }

  auto store = result.value();
  std::vector<tensorstore::Index> shape = { sizeC, sizeZ, sizeY, sizeX };
  auto arr = tensorstore::Array(reinterpret_cast<uint16_t*>(data.get()), shape);
  auto readResult = tensorstore::Read(store, tensorstore::UnownedToShared(arr)).result();
  if (!readResult.ok()) {
    return nullptr;
  }

  ImageXYZC* image = new ImageXYZC(sizeX, sizeY, sizeZ, sizeC, bpp, data.release(), sx, sy, sz, spatialUnits);
  std::shared_ptr<ImageXYZC> sharedImage(image);

  if (meta.contains("channelNames")) {
    std::vector<std::string> channelNames;
    for (auto& item : meta["channelNames"]) {
      channelNames.push_back(item.get<std::string>());
    }
    image->setChannelNames(channelNames);
  }

  meta["lastAccess"] = nowMillis();
  try {
    std::ofstream metaOut(metaPath.string(), std::ios::trunc);
    metaOut << meta.dump(2);
  } catch (...) {
    // best effort
  }

  {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_diskEntries.find(diskCacheId(key));
    if (it != m_diskEntries.end()) {
      it->second.lastAccess = meta["lastAccess"].get<std::uint64_t>();
    }
  }

  return sharedImage;
}

void
CacheManager::storeToDisk(const CacheKey& key, const std::shared_ptr<ImageXYZC>& image, const CacheConfig& config)
{
  if (!image || !config.enableDisk || config.cacheDir.empty()) {
    return;
  }

  std::filesystem::path entryPath = std::filesystem::path(config.cacheDir) / diskCacheId(key);
  std::filesystem::path dataPath = entryPath / "data.zarr";
  std::filesystem::path metaPath = entryPath / "meta.json";
  std::error_code ec;
  std::filesystem::create_directories(entryPath, ec);

  std::uint32_t sizeX = static_cast<std::uint32_t>(image->sizeX());
  std::uint32_t sizeY = static_cast<std::uint32_t>(image->sizeY());
  std::uint32_t sizeZ = static_cast<std::uint32_t>(image->sizeZ());
  std::uint32_t sizeC = static_cast<std::uint32_t>(image->sizeC());
  std::uint32_t bpp = static_cast<std::uint32_t>(ImageXYZC::IN_MEMORY_BPP);

  std::vector<tensorstore::Index> shape = { sizeC, sizeZ, sizeY, sizeX };
  std::vector<tensorstore::Index> chunkShape = { 1,
                                                 std::min<tensorstore::Index>(16, sizeZ),
                                                 std::min<tensorstore::Index>(256, sizeY),
                                                 std::min<tensorstore::Index>(256, sizeX) };

  nlohmann::json schema = { { "dtype", "uint16" },
                            { "domain", { { "shape", shape } } },
                            { "chunk_layout",
                              { { "read_chunk", { { "shape", chunkShape } } },
                                { "write_chunk", { { "shape", chunkShape } } } } } };

  auto openFuture = tensorstore::Open<std::uint16_t, 4, tensorstore::ReadWriteMode::read_write>(
    { { "driver", "zarr3" },
      { "kvstore", { { "driver", "file" }, { "path", dataPath.string() } } },
      { "schema", schema } },
    tensorstore::OpenMode::create | tensorstore::OpenMode::open);
  auto result = openFuture.result();
  if (!result.ok()) {
    return;
  }

  auto store = result.value();
  auto arr = tensorstore::Array(reinterpret_cast<uint16_t*>(image->ptr()), shape);
  auto writeResult = tensorstore::Write(tensorstore::UnownedToShared(arr), store).result();
  if (!writeResult.ok()) {
    return;
  }

  std::uint64_t bytes = estimateImageBytes(*image);
  std::uint64_t accessTime = nowMillis();
  nlohmann::json meta = { { "key", keyToString(key) },
                          { "sizeX", sizeX },
                          { "sizeY", sizeY },
                          { "sizeZ", sizeZ },
                          { "sizeC", sizeC },
                          { "bpp", bpp },
                          { "physicalSizeX", image->physicalSizeX() },
                          { "physicalSizeY", image->physicalSizeY() },
                          { "physicalSizeZ", image->physicalSizeZ() },
                          { "spatialUnits", image->spatialUnits() },
                          { "channelNames", channelNamesFromImage(*image) },
                          { "bytes", bytes },
                          { "lastAccess", accessTime } };

  try {
    std::ofstream metaOut(metaPath.string(), std::ios::trunc);
    metaOut << meta.dump(2);
  } catch (...) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_stats.diskWrites++;
    DiskEntry entry;
    entry.path = entryPath.string();
    entry.bytes = bytes;
    entry.lastAccess = accessTime;
    auto it = m_diskEntries.find(diskCacheId(key));
    if (it != m_diskEntries.end()) {
      m_currentDiskBytes -= it->second.bytes;
    }
    m_diskEntries[diskCacheId(key)] = entry;
    m_currentDiskBytes += bytes;
  }

  evictDiskIfNeeded(config, 0);
}

void
CacheManager::loadDiskIndex(const CacheConfig& config)
{
  if (config.cacheDir.empty() || !config.enableDisk) {
    return;
  }

  std::filesystem::path root(config.cacheDir);
  std::error_code ec;
  std::filesystem::create_directories(root, ec);
  if (!std::filesystem::exists(root)) {
    return;
  }

  std::unordered_map<std::string, DiskEntry> entries;
  std::uint64_t totalBytes = 0;

  for (const auto& dirEntry : std::filesystem::directory_iterator(root)) {
    if (!dirEntry.is_directory()) {
      continue;
    }

    std::filesystem::path metaPath = dirEntry.path() / "meta.json";
    if (!std::filesystem::exists(metaPath)) {
      continue;
    }

    try {
      nlohmann::json meta;
      std::ifstream metaFile(metaPath.string());
      metaFile >> meta;
      if (!meta.contains("lastAccess")) {
        continue;
      }

      DiskEntry entry;
      entry.path = dirEntry.path().string();
      entry.lastAccess = meta["lastAccess"].get<std::uint64_t>();
      if (meta.contains("bytes")) {
        entry.bytes = meta["bytes"].get<std::uint64_t>();
      } else {
        entry.bytes = directorySizeBytes(entry.path);
      }

      std::string id = dirEntry.path().filename().string();
      entries[id] = entry;
      totalBytes += entry.bytes;
    } catch (...) {
      continue;
    }
  }

  std::lock_guard<std::mutex> lock(m_mutex);
  m_diskEntries = std::move(entries);
  m_currentDiskBytes = totalBytes;
}

void
CacheManager::evictDiskIfNeeded(const CacheConfig& config, std::uint64_t incomingBytes)
{
  if (!config.enableDisk || config.maxDiskBytes == 0) {
    return;
  }

  while (true) {
    std::string oldestKey;
    std::uint64_t oldestAccess = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t currentBytes = 0;
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      currentBytes = m_currentDiskBytes;
      if ((currentBytes + incomingBytes) <= config.maxDiskBytes || m_diskEntries.empty()) {
        break;
      }
      for (const auto& item : m_diskEntries) {
        if (item.second.lastAccess < oldestAccess) {
          oldestAccess = item.second.lastAccess;
          oldestKey = item.first;
        }
      }
    }

    if (oldestKey.empty()) {
      break;
    }

    std::string removePath;
    std::uint64_t removedBytes = 0;
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      auto it = m_diskEntries.find(oldestKey);
      if (it == m_diskEntries.end()) {
        continue;
      }
      removePath = it->second.path;
      removedBytes = it->second.bytes;
      m_diskEntries.erase(it);
      if (m_currentDiskBytes >= removedBytes) {
        m_currentDiskBytes -= removedBytes;
      } else {
        m_currentDiskBytes = 0;
      }
    }

    if (!removePath.empty()) {
      std::error_code ec;
      std::filesystem::remove_all(removePath, ec);
    }
  }
}

std::uint64_t
CacheManager::directorySizeBytes(const std::string& path) const
{
  std::uint64_t total = 0;
  std::error_code ec;
  for (const auto& entry : std::filesystem::recursive_directory_iterator(path, ec)) {
    if (entry.is_regular_file()) {
      total += static_cast<std::uint64_t>(entry.file_size());
    }
  }
  return total;
}
