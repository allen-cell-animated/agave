#pragma once

#include "CacheConfig.h"
#include "IFileReader.h"

#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

class ImageXYZC;

namespace renderlib {

struct CacheKey
{
  std::string filepath;
  std::string subpath;
  int scene = 0;
  std::uint32_t time = 0;
  std::vector<std::uint32_t> channels;
  std::uint32_t minx = 0;
  std::uint32_t maxx = 0;
  std::uint32_t miny = 0;
  std::uint32_t maxy = 0;
  std::uint32_t minz = 0;
  std::uint32_t maxz = 0;
  bool isImageSequence = false;

  bool operator==(const CacheKey& other) const;
};

struct CacheKeyHash
{
  std::size_t operator()(const CacheKey& key) const;
};

class CacheManager
{
public:
  struct CacheStats
  {
    std::uint64_t ramHits = 0;
    std::uint64_t diskHits = 0;
    std::uint64_t misses = 0;
    std::uint64_t diskWrites = 0;
  };

  static CacheManager& instance();

  void setConfig(const CacheConfig& config);
  CacheConfig getConfig() const;

  std::shared_ptr<ImageXYZC> findImage(const LoadSpec& loadSpec);
  void storeImage(const LoadSpec& loadSpec, const std::shared_ptr<ImageXYZC>& image);
  void clear();
  void clearDiskCache();

  CacheStats getStats() const;
  void resetStats();

private:
  CacheManager() = default;

  CacheKey makeKey(const LoadSpec& loadSpec) const;
  std::string keyToString(const CacheKey& key) const;
  std::string diskCacheId(const CacheKey& key) const;
  std::uint64_t estimateImageBytes(const ImageXYZC& image) const;
  void touchEntry(std::list<CacheKey>::iterator it);
  void evictIfNeeded(std::uint64_t incomingBytes);
  void storeImageInMemory(const CacheKey& key, const std::shared_ptr<ImageXYZC>& image);

  std::shared_ptr<ImageXYZC> loadFromDisk(const CacheKey& key, const CacheConfig& config);
  void storeToDisk(const CacheKey& key, const std::shared_ptr<ImageXYZC>& image, const CacheConfig& config);
  void loadDiskIndex(const CacheConfig& config);
  void evictDiskIfNeeded(const CacheConfig& config, std::uint64_t incomingBytes);
  std::uint64_t directorySizeBytes(const std::string& path) const;

  mutable std::mutex m_mutex;
  CacheConfig m_config;
  std::uint64_t m_currentRamBytes = 0;
  std::list<CacheKey> m_lruKeys;

  struct CacheEntry
  {
    std::shared_ptr<ImageXYZC> image;
    std::uint64_t bytes = 0;
    std::list<CacheKey>::iterator lruIt;
  };

  std::unordered_map<CacheKey, CacheEntry, CacheKeyHash> m_entries;

  struct DiskEntry
  {
    std::string path;
    std::uint64_t bytes = 0;
    std::uint64_t lastAccess = 0;
  };

  std::unordered_map<std::string, DiskEntry> m_diskEntries;
  std::uint64_t m_currentDiskBytes = 0;
  std::string m_diskIndexRoot;

  CacheStats m_stats;
};

} // namespace renderlib
