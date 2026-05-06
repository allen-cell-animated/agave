#include "FileReaderCzi.h"

#include "BoundingBox.h"
#include "ImageXYZC.h"
#include "Logging.h"
#include "VolumeDimensions.h"

#include <libCZI/Src/libCZI/libCZI.h>

#include "pugixml/pugixml.hpp"

#include <filesystem>

#include <atomic>
#include <chrono>
#include <codecvt>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <set>
#include <thread>
#include <utility>
#include <vector>

namespace {
bool
isHttpUrl(const std::string& filepath)
{
  return filepath.rfind("http://", 0) == 0 || filepath.rfind("https://", 0) == 0;
}

// Initialize libCZI's stream factory exactly once. This sets up the global
// state required by the curl-based HTTP/HTTPS input stream (when libCZI was
// built with curl support). Safe to call from any thread.
void
ensureStreamsFactoryInitialized()
{
  static std::once_flag s_initFlag;
  std::call_once(s_initFlag, []() { libCZI::StreamsFactory::Initialize(); });
}

// Populate sensible HTTP options for libCZI's curl-based input stream.
void
applyHttpStreamProperties(libCZI::StreamsFactory::CreateStreamInfo& streamInfo)
{
  using SP = libCZI::StreamsFactory::StreamProperties;
  using P = libCZI::StreamsFactory::Property;
  streamInfo.property_bag[SP::kCurlHttp_FollowLocation] = P(true);
  streamInfo.property_bag[SP::kCurlHttp_Timeout] = P(static_cast<std::int32_t>(120));
  streamInfo.property_bag[SP::kCurlHttp_ConnectTimeout] = P(static_cast<std::int32_t>(30));
  streamInfo.property_bag[SP::kCurlHttp_UserAgent] = P(std::string("agave"));
}

// A per-URL shared cache of byte windows. Multiple CoalescingStream
// instances for the same URL share one of these so the (large, identical)
// reads done by ICZIReader::Open() - file header, metadata segment, and
// the CZI subblock-directory body - are fetched once and reused across
// concurrent worker streams.
class SharedStreamCache
{
public:
  static constexpr size_t kDefaultMaxWindows = 8;

  // Returns true and copies into `dst` if a cached window fully contains
  // [offset, offset+size). LRU-touches on hit.
  bool tryGet(std::uint64_t offset, std::uint64_t size, std::uint8_t* dst)
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto it = m_windows.begin(); it != m_windows.end(); ++it) {
      const Window& w = **it;
      if (offset >= w.offset && offset + size <= w.offset + w.data.size()) {
        std::memcpy(dst, w.data.data() + (offset - w.offset), static_cast<size_t>(size));
        if (it != m_windows.begin()) {
          auto p = std::move(*it);
          m_windows.erase(it);
          m_windows.push_front(std::move(p));
        }
        return true;
      }
    }
    return false;
  }

  // Coordinate concurrent fetches of the same (offset, size) range.
  //
  // Returns true if the caller should perform the fetch (and later call
  // endFetch). Returns false if another caller was already fetching that
  // exact range; in that case beginFetch blocks until the in-flight fetch
  // completes and the caller should retry tryGet first.
  //
  // Without this, all 8 worker Open()s race to fetch the (large) CZI
  // subblock directory simultaneously, since insert happens only at the
  // end of the first fetch.
  bool beginFetch(std::uint64_t offset, std::uint64_t size)
  {
    std::unique_lock<std::mutex> lock(m_mutex);
    auto matches = [&]() {
      for (auto& p : m_inflight) {
        if (p.first == offset && p.second == size) {
          return true;
        }
      }
      return false;
    };
    if (matches()) {
      m_cv.wait(lock, [&]() { return !matches(); });
      return false;
    }
    m_inflight.emplace_back(offset, size);
    return true;
  }

  // Inserts a window (if non-empty) and clears the in-flight marker.
  void endFetch(std::uint64_t offset, std::uint64_t size, std::vector<std::uint8_t>&& data)
  {
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      if (!data.empty()) {
        while (m_windows.size() >= kDefaultMaxWindows) {
          m_windows.pop_back();
        }
        auto w = std::make_unique<Window>();
        w->offset = offset;
        w->data = std::move(data);
        m_windows.push_front(std::move(w));
      }
      for (auto it = m_inflight.begin(); it != m_inflight.end(); ++it) {
        if (it->first == offset && it->second == size) {
          m_inflight.erase(it);
          break;
        }
      }
    }
    m_cv.notify_all();
  }

private:
  struct Window
  {
    std::uint64_t offset;
    std::vector<std::uint8_t> data;
  };
  std::deque<std::unique_ptr<Window>> m_windows;
  std::vector<std::pair<std::uint64_t, std::uint64_t>> m_inflight;
  std::mutex m_mutex;
  std::condition_variable m_cv;
};

// Process-wide registry mapping a file identifier (URL) to a SharedStreamCache.
// Strong-ref so the cache survives across the loadNumScenes / loadDimensions /
// loadFromFile sequence, which today destroys and recreates ScopedCziReader
// instances between phases. Bounded by an LRU over URLs.
std::shared_ptr<SharedStreamCache>
acquireSharedCache(const std::string& filepath)
{
  static constexpr size_t kMaxUrlEntries = 4;
  static std::mutex s_mutex;
  static std::deque<std::pair<std::string, std::shared_ptr<SharedStreamCache>>> s_caches;
  std::lock_guard<std::mutex> lock(s_mutex);
  for (auto it = s_caches.begin(); it != s_caches.end(); ++it) {
    if (it->first == filepath) {
      auto sp = it->second;
      if (it != s_caches.begin()) {
        auto entry = std::move(*it);
        s_caches.erase(it);
        s_caches.push_front(std::move(entry));
      }
      return sp;
    }
  }
  auto sp = std::make_shared<SharedStreamCache>();
  s_caches.push_front({ filepath, sp });
  while (s_caches.size() > kMaxUrlEntries) {
    s_caches.pop_back();
  }
  return sp;
}

// Wraps a libCZI IStream and coalesces sequential reads at adjacent offsets
// into a single fetch. Each call to ICZIReader::ReadSubBlock issues 5
// sequential Read()s (subblock header, continuation header, XML metadata,
// pixel data, attachment) at adjacent offsets; ICZIReader::Open issues
// several reads at the file header, metadata segment, and subblock directory.
// On a high-RTT remote stream those round-trips dominate. By prefetching a
// sizeable window anchored at the requested offset on each cache miss, all
// follow-up reads inside one logical operation become cache hits.
//
// Multiple CoalescingStreams for the same URL share a SharedStreamCache so
// the (identical, large) bootstrap reads done by every parallel worker's
// Open() are fetched once and reused.
class CoalescingStream : public libCZI::IStream
{
public:
  // Window size aims to cover one subblock's combined header + metadata +
  // pixel data + attachment in a single HTTP request. Smaller wastes
  // round-trips; larger wastes bandwidth on per-Open and tail-of-file reads.
  // 14 MB sized for a typical 13.2 MB subblock stride; tweak if subblocks
  // get larger.
  static constexpr std::uint64_t kDefaultWindowSize = 14 * 1024 * 1024;

  CoalescingStream(std::shared_ptr<libCZI::IStream> inner,
                   std::shared_ptr<SharedStreamCache> sharedCache,
                   std::uint64_t windowSize = kDefaultWindowSize)
    : m_inner(std::move(inner))
    , m_cache(std::move(sharedCache))
    , m_windowSize(windowSize)
  {
  }

  void Read(std::uint64_t offset, void* pv, std::uint64_t size, std::uint64_t* ptrBytesRead) override
  {
    if (size == 0) {
      if (ptrBytesRead) {
        *ptrBytesRead = 0;
      }
      return;
    }
    auto* dst = static_cast<std::uint8_t*>(pv);
    std::uint64_t fetchSize = std::max<std::uint64_t>(size, m_windowSize);

    while (true) {
      // Fast path: served entirely from the shared cache.
      if (m_cache->tryGet(offset, size, dst)) {
        if (ptrBytesRead) {
          *ptrBytesRead = size;
        }
        return;
      }

      // Single-flight: if another stream is already fetching this exact
      // range, wait for it and re-check the cache. This dedupes the
      // bootstrap reads done by every parallel worker's Open().
      if (!m_cache->beginFetch(offset, fetchSize)) {
        continue;
      }

      // Miss: fetch a window anchored at `offset`, sized to cover this read
      // plus a prefetch margin. Always cache the bytes so siblings hit.
      std::vector<std::uint8_t> buf(static_cast<size_t>(fetchSize));
      std::uint64_t got = directRead(offset, buf.data(), fetchSize);
      buf.resize(static_cast<size_t>(got));

      std::uint64_t served = std::min<std::uint64_t>(got, size);
      if (served > 0) {
        std::memcpy(dst, buf.data(), static_cast<size_t>(served));
      }
      if (ptrBytesRead) {
        *ptrBytesRead = served;
      }
      m_cache->endFetch(offset, fetchSize, std::move(buf));
      return;
    }
  }

private:
  // Performs one Read on the inner stream and logs its timing.
  std::uint64_t directRead(std::uint64_t offset, void* dst, std::uint64_t requested)
  {
    auto t0 = std::chrono::steady_clock::now();
    std::uint64_t got = 0;
    m_inner->Read(offset, dst, requested, &got);
    auto t1 = std::chrono::steady_clock::now();
    LOG_DEBUG << "CoalescingStream inner Read offset=" << offset << " requested=" << requested << " got=" << got
              << " in " << std::chrono::duration<double, std::milli>(t1 - t0).count() << "ms";
    return got;
  }

  std::shared_ptr<libCZI::IStream> m_inner;
  std::shared_ptr<SharedStreamCache> m_cache;
  std::uint64_t m_windowSize;
};
} // namespace

class ScopedCziReader
{
public:
  ScopedCziReader(const std::string& filepath)
  {
    std::shared_ptr<libCZI::IStream> stream;
    if (isHttpUrl(filepath)) {
      ensureStreamsFactoryInitialized();
      libCZI::StreamsFactory::CreateStreamInfo streamInfo;
      streamInfo.class_name = "curl_http_inputstream";
      applyHttpStreamProperties(streamInfo);
      stream = libCZI::StreamsFactory::CreateStream(streamInfo, filepath);
      if (!stream) {
        LOG_ERROR << "Failed to create HTTP stream for " << filepath
                  << " (libCZI may have been built without curl support)";
        throw std::runtime_error("Failed to create HTTP stream for CZI URL");
      }
      // Wrap with a coalescing/caching layer that shares its cache across
      // all streams for this URL. Each parallel worker's Open() then
      // re-reads the (large) directory body from RAM instead of HTTP.
      stream = std::make_shared<CoalescingStream>(std::move(stream), acquireSharedCache(filepath));
    } else {
      std::filesystem::path fpath(filepath);
      const std::wstring widestr = fpath.wstring();
      stream = libCZI::CreateStreamFromFile(widestr.c_str());
    }

    m_reader = libCZI::CreateCZIReader();
    m_reader->Open(stream);
  }
  ~ScopedCziReader()
  {
    if (m_reader) {
      m_reader->Close();
    }
  }
  std::shared_ptr<libCZI::ICZIReader> reader() { return m_reader; }

protected:
  std::shared_ptr<libCZI::ICZIReader> m_reader;
};

// Per-instance cache of an open CZI reader plus its statistics. Reusing this
// across loadDimensions/loadNumScenes/loadMultiscaleDims/loadFromFile saves
// one HTTP round-trip (Open) + one statistics scan per redundant call.
struct CziReaderState
{
  std::string filepath;
  std::unique_ptr<ScopedCziReader> reader;
  libCZI::SubBlockStatistics statistics;
};

FileReaderCzi::FileReaderCzi(const std::string& filepath) {}

FileReaderCzi::~FileReaderCzi() {}

namespace {
CziReaderState&
openOrReuse(std::unique_ptr<CziReaderState>& state, const std::string& filepath)
{
  if (state && state->filepath == filepath && state->reader) {
    return *state;
  }
  auto fresh = std::make_unique<CziReaderState>();
  fresh->filepath = filepath;
  fresh->reader = std::make_unique<ScopedCziReader>(filepath);
  fresh->statistics = fresh->reader->reader()->GetStatistics();
  state = std::move(fresh);
  return *state;
}
} // namespace

libCZI::IntRect
getSceneYXSize(libCZI::SubBlockStatistics& statistics, int sceneIndex = 0)
{
  bool hasScene = statistics.dimBounds.IsValid(libCZI::DimensionIndex::S);
  if (hasScene) {
    int sStart(0), sSize(0);
    statistics.dimBounds.TryGetInterval(libCZI::DimensionIndex::S, &sStart, &sSize);
    if (sceneIndex >= sStart && (sStart + sSize - 1) >= sceneIndex && !statistics.sceneBoundingBoxes.empty()) {
      return statistics.sceneBoundingBoxes[sceneIndex].boundingBoxLayer0;
    }
  } else {
    return statistics.boundingBoxLayer0Only;
  }
  return statistics.boundingBoxLayer0Only;
}

pugi_agave::xml_node
firstChild(pugi_agave::xml_node& el, std::string tag)
{
  pugi_agave::xml_node child = el.child(tag.c_str());
  if (!child) {
    LOG_ERROR << "No " << tag << " element in xml";
  }
  return child;
}

bool
readCziDimensions(const std::shared_ptr<libCZI::ICZIReader>& reader,
                  const std::string filepath,
                  libCZI::SubBlockStatistics& statistics,
                  VolumeDimensions& dims,
                  uint32_t scene)
{
  // metadata xml
  auto mds = reader->ReadMetadataSegment();
  std::shared_ptr<libCZI::ICziMetadata> md = mds->CreateMetaFromMetadataSegment();
  std::shared_ptr<libCZI::ICziMultiDimensionDocumentInfo> docinfo = md->GetDocumentInfo();

  libCZI::ScalingInfoEx scalingInfo = docinfo->GetScalingInfoEx();
  // convert meters to microns?
  dims.physicalSizeX = (float)(scalingInfo.scaleX * 1000000.0);
  dims.physicalSizeY = (float)(scalingInfo.scaleY * 1000000.0);
  dims.physicalSizeZ = (float)(scalingInfo.scaleZ * 1000000.0);
  // just select the x unit for our units.
  using convert_typeX = std::codecvt_utf8<wchar_t>;
  std::wstring_convert<convert_typeX, wchar_t> converterX;
  dims.spatialUnits = VolumeDimensions::sanitizeUnitsString(converterX.to_bytes(scalingInfo.defaultUnitFormatX));

  // get all dimension bounds and enumerate.
  // I am making an assumption here that each scene has the same Z C and T dimensions.
  // If acquisition ended early, then the data might not be complete and that will have to be caught later.
  statistics.dimBounds.EnumValidDimensions([&](libCZI::DimensionIndex dimensionIndex, int start, int size) -> bool {
    switch (dimensionIndex) {
      case libCZI::DimensionIndex::Z:
        dims.sizeZ = size;
        break;
      case libCZI::DimensionIndex::C:
        dims.sizeC = size;
        break;
      case libCZI::DimensionIndex::T:
        dims.sizeT = size;
        break;
      default:
        break;
    }
    return true;
  });

  libCZI::IntRect planebox = getSceneYXSize(statistics, scene);
  dims.sizeX = planebox.w;
  dims.sizeY = planebox.h;

  std::string xml = md->GetXml();

  pugi_agave::xml_document czixml;
  pugi_agave::xml_parse_result parseOk = czixml.load_string(xml.c_str());
  if (!parseOk) {
    LOG_ERROR << "Bad CZI xml metadata content";
    return false;
  }

  pugi_agave::xml_node imageDocumentEl = firstChild(czixml, "ImageDocument");
  if (!imageDocumentEl) {
    return false;
  }
  pugi_agave::xml_node metadataEl = firstChild(imageDocumentEl, "Metadata");
  if (!metadataEl) {
    return false;
  }
  pugi_agave::xml_node informationEl = firstChild(metadataEl, "Information");
  if (!informationEl) {
    return false;
  }
  pugi_agave::xml_node imageEl = firstChild(informationEl, "Image");
  if (!imageEl) {
    return false;
  }
  pugi_agave::xml_node dimensionsEl = firstChild(imageEl, "Dimensions");
  if (!dimensionsEl) {
    return false;
  }
  pugi_agave::xml_node channelsEl = firstChild(dimensionsEl, "Channels");
  if (!channelsEl) {
    return false;
  }
  std::vector<std::string> channelNames;
  for (pugi_agave::xml_node node : channelsEl.children("Channel")) {
    channelNames.push_back(node.attribute("Name").value());
  }
  dims.channelNames = channelNames;

  libCZI::SubBlockInfo info;
  bool ok = reader->TryGetSubBlockInfoOfArbitrarySubBlockInChannel(0, info);
  if (ok) {
    switch (info.pixelType) {
      case libCZI::PixelType::Gray8:
        dims.bitsPerPixel = 8;
        break;
      case libCZI::PixelType::Gray16:
        dims.bitsPerPixel = 16;
        break;
      case libCZI::PixelType::Gray32Float:
        dims.bitsPerPixel = 32;
        break;
      case libCZI::PixelType::Bgr24:
        dims.bitsPerPixel = 24;
        break;
      case libCZI::PixelType::Bgr48:
        dims.bitsPerPixel = 48;
        break;
      case libCZI::PixelType::Bgr96Float:
        dims.bitsPerPixel = 96;
        break;
      default:
        dims.bitsPerPixel = 0;
        return false;
    }
  } else {
    return false;
  }

  return dims.validate();
}

// Copy pixel data from a libCZI bitmap (sized exactly volumeDims.sizeX/sizeY)
// into the destination buffer at dataPtr (uint16 packed, sizeX*2 bytes per row).
// Promotes 8-bit to 16-bit; passes 16-bit through with stride conversion.
bool
copyBitmapToDest(const std::shared_ptr<libCZI::IBitmapData>& bitmap,
                 const VolumeDimensions& volumeDims,
                 uint8_t* dataPtr)
{
  libCZI::IntSize size = bitmap->GetSize();
  libCZI::ScopedBitmapLockerSP lckScoped{ bitmap };
  assert(lckScoped.ptrDataRoi == lckScoped.ptrData);
  assert(volumeDims.sizeX == size.w);
  assert(volumeDims.sizeY == size.h);
  size_t bytesPerRow = size.w * 2; // destination stride
  if (volumeDims.bitsPerPixel == 16) {
    assert(lckScoped.stride >= size.w * 2);
    for (std::uint32_t y = 0; y < size.h; ++y) {
      const std::uint8_t* ptrLine = ((const std::uint8_t*)lckScoped.ptrDataRoi) + y * lckScoped.stride;
      memcpy(dataPtr + (bytesPerRow * y), ptrLine, bytesPerRow);
    }
  } else if (volumeDims.bitsPerPixel == 8) {
    assert(lckScoped.stride >= size.w);
    for (std::uint32_t y = 0; y < size.h; ++y) {
      const std::uint8_t* ptrLine = ((const std::uint8_t*)lckScoped.ptrDataRoi) + y * lckScoped.stride;
      uint16_t* destLine = reinterpret_cast<uint16_t*>(dataPtr + (bytesPerRow * y));
      for (size_t x = 0; x < size.w; ++x) {
        *destLine++ = *(ptrLine + x);
      }
    }
  }
  return true;
}

// DANGER: assumes dataPtr has enough space allocated!!!!
bool
readCziPlane(const std::shared_ptr<libCZI::ICZIReader>& reader,
             const libCZI::IntRect& planeRect,
             const libCZI::CDimCoordinate& planeCoord,
             const VolumeDimensions& volumeDims,
             const libCZI::ISingleChannelPyramidLayerTileAccessor::Options* options,
             uint8_t* dataPtr)
{
  auto accessor = reader->CreateSingleChannelPyramidLayerTileAccessor();

  libCZI::ISingleChannelPyramidLayerTileAccessor::PyramidLayerInfo pyrLyrInfo;
  pyrLyrInfo.minificationFactor = 1;
  pyrLyrInfo.pyramidLayerNo = 0;

  auto bitmap = accessor->Get(planeRect, &planeCoord, pyrLyrInfo, options);
  return copyBitmapToDest(bitmap, volumeDims, dataPtr);
}

// Fast-path: read a single subblock directly and copy its pixels into the
// destination. Avoids the accessor's compositing overhead and an extra bitmap
// allocation. The caller has already verified the subblock's physical size
// matches the destination plane.
bool
readCziPlaneDirect(const std::shared_ptr<libCZI::ICZIReader>& reader,
                   int subblockIndex,
                   const VolumeDimensions& volumeDims,
                   uint8_t* dataPtr)
{
  auto subblock = reader->ReadSubBlock(subblockIndex);
  if (!subblock) {
    return false;
  }
  auto bitmap = subblock->CreateBitmap();
  return copyBitmapToDest(bitmap, volumeDims, dataPtr);
}

uint32_t
FileReaderCzi::loadNumScenes(const std::string& filepath)
{
  try {
    auto& state = openOrReuse(m_state, filepath);
    auto& statistics = state.statistics;
    int sceneStart = 0;
    int sceneSize = 0;
    bool scenesDefined = statistics.dimBounds.TryGetInterval(libCZI::DimensionIndex::S, &sceneStart, &sceneSize);
    if (!scenesDefined) {
      // assume just one?
      return 1;
    }
    return sceneSize;

  } catch (std::exception& e) {
    LOG_ERROR << e.what();
    LOG_ERROR << "Failed to read " << filepath;
    return 0;
  } catch (...) {
    LOG_ERROR << "Failed to read " << filepath;
    return 0;
  }
  return 0;
}

VolumeDimensions
FileReaderCzi::loadDimensions(const std::string& filepath, uint32_t scene)
{
  VolumeDimensions dims;
  try {
    auto& state = openOrReuse(m_state, filepath);
    std::shared_ptr<libCZI::ICZIReader> cziReader = state.reader->reader();
    auto& statistics = state.statistics;

    bool dims_ok = readCziDimensions(cziReader, filepath, statistics, dims, scene);
    if (!dims_ok) {
      return VolumeDimensions();
    }

    return dims;

  } catch (std::exception& e) {
    LOG_ERROR << e.what();
    LOG_ERROR << "Failed to read " << filepath;
    return VolumeDimensions();
  } catch (...) {
    LOG_ERROR << "Failed to read " << filepath;
    return VolumeDimensions();
  }
}

std::shared_ptr<ImageXYZC>
FileReaderCzi::loadFromFile(const LoadSpec& loadSpec)
{
  std::string filepath = loadSpec.filepath;
  uint32_t scene = loadSpec.scene;
  uint32_t time = loadSpec.time;
  VolumeDimensions outDims;

  std::shared_ptr<ImageXYZC> emptyimage;

  auto tStart = std::chrono::high_resolution_clock::now();

  try {
    auto& state = openOrReuse(m_state, filepath);
    std::shared_ptr<libCZI::ICZIReader> cziReader = state.reader->reader();
    auto& statistics = state.statistics;

    VolumeDimensions dims;
    bool dims_ok = readCziDimensions(cziReader, filepath, statistics, dims, scene);
    if (!dims_ok) {
      return emptyimage;
    }
    int startT = 0, sizeT = 0;
    int startC = 0, sizeC = 0;
    int startZ = 0, sizeZ = 0;
    int startS = 0, sizeS = 0;
    bool hasT = statistics.dimBounds.TryGetInterval(libCZI::DimensionIndex::T, &startT, &sizeT);
    bool hasZ = statistics.dimBounds.TryGetInterval(libCZI::DimensionIndex::Z, &startZ, &sizeZ);
    bool hasC = statistics.dimBounds.TryGetInterval(libCZI::DimensionIndex::C, &startC, &sizeC);
    bool hasS = statistics.dimBounds.TryGetInterval(libCZI::DimensionIndex::S, &startS, &sizeS);

    if (!hasZ) {
      LOG_ERROR << "AGAVE can only read zstack volume data";
      return emptyimage;
    }
    if (dims.sizeC != sizeC || !hasC) {
      LOG_ERROR << "Inconsistent Channel count in czi file";
      return emptyimage;
    }

    size_t nch = loadSpec.channels.empty() ? dims.sizeC : loadSpec.channels.size();

    size_t planesize = (size_t)dims.sizeX * (size_t)dims.sizeY * (size_t)dims.bitsPerPixel / 8;
    uint8_t* data = new uint8_t[planesize * dims.sizeZ * nch];
    memset(data, 0, planesize * dims.sizeZ * nch);

    // stash it here in case of early exit, it will be deleted
    std::unique_ptr<uint8_t[]> smartPtr(data);

    uint8_t* destptr = data;

    // now ready to read channels one by one.
    libCZI::IntRect planeRect;
    if (hasS) {
      planeRect = statistics.sceneBoundingBoxes[startS + scene].boundingBoxLayer0;
    } else {
      planeRect = statistics.boundingBoxLayer0Only;
    }

    auto buildOptions = [&]() {
      libCZI::ISingleChannelPyramidLayerTileAccessor::Options o;
      o.Clear();
      if (hasS) {
        std::wstringstream wss;
        wss << scene;
        o.sceneFilter = libCZI::Utils::IndexSetFromString(wss.str());
      }
      return o;
    };

    // Build a (channelAbs, sliceAbs) -> subblockIndex map for a fast direct
    // read path that bypasses the accessor when a plane is backed by exactly
    // one subblock that matches the destination size (the common non-mosaic
    // case). Multi-subblock planes fall back to the accessor.
    auto tIdxStart = std::chrono::high_resolution_clock::now();
    // Encode (channelAbs * dims.sizeZ + sliceAbs) directly; both are small.
    std::vector<std::vector<int>> subblocksByPlane(static_cast<size_t>(sizeC) * static_cast<size_t>(dims.sizeZ));
    std::vector<libCZI::IntSize> physSizesByPlane(subblocksByPlane.size(), libCZI::IntSize{ 0, 0 });
    {
      libCZI::CDimCoordinate filterCoord;
      if (hasT) {
        filterCoord.Set(libCZI::DimensionIndex::T, time + startT);
      }
      if (hasS) {
        filterCoord.Set(libCZI::DimensionIndex::S, startS + scene);
      }
      cziReader->EnumSubset(&filterCoord, nullptr, true, [&](int idx, const libCZI::SubBlockInfo& info) -> bool {
        int c = 0;
        int z = 0;
        if (!info.coordinate.TryGetPosition(libCZI::DimensionIndex::C, &c)) {
          c = startC;
        }
        if (!info.coordinate.TryGetPosition(libCZI::DimensionIndex::Z, &z)) {
          z = startZ;
        }
        size_t cIdx = static_cast<size_t>(c - startC);
        size_t zIdx = static_cast<size_t>(z - startZ);
        if (cIdx < static_cast<size_t>(sizeC) && zIdx < static_cast<size_t>(dims.sizeZ)) {
          size_t key = cIdx * static_cast<size_t>(dims.sizeZ) + zIdx;
          subblocksByPlane[key].push_back(idx);
          physSizesByPlane[key] = info.physicalSize;
        }
        return true;
      });
    }
    auto tIdxEnd = std::chrono::high_resolution_clock::now();

    // Build the flat job list once; both the serial and parallel paths consume it.
    struct Job
    {
      uint32_t channelIdx;
      uint32_t channelToLoad;
      uint32_t slice;
      int subblockIndex; // -1 if no direct fast path; use accessor instead
    };
    std::vector<Job> jobs;
    jobs.reserve(static_cast<size_t>(nch) * dims.sizeZ);
    size_t directCount = 0;
    size_t accessorCount = 0;
    for (uint32_t channel = 0; channel < nch; ++channel) {
      uint32_t channelToLoad = channel;
      if (!loadSpec.channels.empty()) {
        channelToLoad = loadSpec.channels[channel];
      }
      for (uint32_t slice = 0; slice < dims.sizeZ; ++slice) {
        int subblockIndex = -1;
        size_t cIdx = static_cast<size_t>(channelToLoad);
        if (cIdx < static_cast<size_t>(sizeC)) {
          size_t key = cIdx * static_cast<size_t>(dims.sizeZ) + static_cast<size_t>(slice);
          const auto& list = subblocksByPlane[key];
          const auto& phys = physSizesByPlane[key];
          if (list.size() == 1 && phys.w == static_cast<std::uint32_t>(dims.sizeX) &&
              phys.h == static_cast<std::uint32_t>(dims.sizeY)) {
            subblockIndex = list.front();
            ++directCount;
          } else {
            ++accessorCount;
          }
        } else {
          ++accessorCount;
        }
        jobs.push_back({ channel, channelToLoad, slice, subblockIndex });
      }
    }
    LOG_DEBUG << "CZI subblock index built in "
              << std::chrono::duration<double, std::milli>(tIdxEnd - tIdxStart).count() << "ms (" << directCount
              << " direct, " << accessorCount << " via accessor)";

    auto runJob = [&](const std::shared_ptr<libCZI::ICZIReader>& workerReader,
                      const libCZI::ISingleChannelPyramidLayerTileAccessor::Options& opts,
                      const Job& job) -> bool {
      uint8_t* destptr = data + planesize * (job.channelIdx * dims.sizeZ + job.slice);
      if (job.subblockIndex >= 0) {
        return readCziPlaneDirect(workerReader, job.subblockIndex, dims, destptr);
      }
      libCZI::CDimCoordinate planeCoord{ { libCZI::DimensionIndex::Z, (int)job.slice + startZ } };
      if (hasC) {
        planeCoord.Set(libCZI::DimensionIndex::C, (int)job.channelToLoad + startC);
      }
      if (hasT) {
        planeCoord.Set(libCZI::DimensionIndex::T, time + startT);
      }
      // since scene tiles can not overlap, passing the scene bounding box in to readCziPlane is enough produce the
      // scene, and I don't need to add Scene to the planeCoord.
      return readCziPlane(workerReader, planeRect, planeCoord, dims, &opts, destptr);
    };

    auto tReadStart = std::chrono::high_resolution_clock::now();
    if (!isHttpUrl(filepath)) {
      // Local files: keep serial reads (single mmap'd stream is plenty fast,
      // and parallel reads on one disk can hurt rather than help).
      auto opts = buildOptions();
      for (const auto& job : jobs) {
        if (!runJob(cziReader, opts, job)) {
          return emptyimage;
        }
      }
    } else {
      // Remote URL: parallelize across N independent libCZI readers, each with
      // its own curl handle / TCP+TLS connection. libCZI's curl IStream
      // serializes Read() per-stream, so true parallelism requires distinct
      // streams.
      unsigned hwc = std::thread::hardware_concurrency();
      unsigned numWorkers = std::min<unsigned>(8u, hwc > 1u ? hwc : 1u);
      if (numWorkers < 1u) {
        numWorkers = 1u;
      }
      if (jobs.size() < numWorkers) {
        numWorkers = static_cast<unsigned>(jobs.size());
      }
      auto tOpenStart = std::chrono::high_resolution_clock::now();
      // Each Open() does a remote walk of the CZI directory and is expensive.
      // Rather than opening all extra readers up front and then starting
      // workers, we overlap: worker 0 reuses the already-cached reader and
      // starts pulling jobs immediately, while extra-worker threads each
      // open() and only then begin pulling jobs. Total wall-clock becomes
      // roughly max(open_time, single_worker_remaining_planes) instead of
      // open_time + total_plane_work.
      const unsigned numExtra = numWorkers > 0 ? numWorkers - 1 : 0;

      std::atomic<size_t> nextJob{ 0 };
      std::atomic<bool> failed{ false };
      std::mutex errorMutex;
      std::string errorMessage;
      std::atomic<unsigned> openedExtra{ 0 };

      auto recordError = [&](const std::string& msg) {
        if (!failed.exchange(true)) {
          std::lock_guard<std::mutex> lock(errorMutex);
          errorMessage = msg;
        }
      };

      auto runWorker = [&](const std::shared_ptr<libCZI::ICZIReader>& workerReader) {
        auto opts = buildOptions();
        while (!failed.load(std::memory_order_relaxed)) {
          size_t idx = nextJob.fetch_add(1, std::memory_order_relaxed);
          if (idx >= jobs.size()) {
            return;
          }
          try {
            if (!runJob(workerReader, opts, jobs[idx])) {
              recordError("readCziPlane failed");
              return;
            }
          } catch (const std::exception& e) {
            recordError(e.what());
            return;
          } catch (...) {
            recordError("unknown exception in CZI worker");
            return;
          }
        }
      };

      // Holds an extra reader for the lifetime of the worker thread that
      // owns it; freed when threads are joined below.
      std::vector<std::unique_ptr<ScopedCziReader>> extraReaders(numExtra);

      std::vector<std::thread> threads;
      threads.reserve(numExtra);
      for (unsigned w = 0; w < numExtra; ++w) {
        threads.emplace_back([&, w]() {
          // Bail out cheaply if everything is already done before this
          // reader even finished opening.
          if (failed.load(std::memory_order_relaxed) || nextJob.load(std::memory_order_relaxed) >= jobs.size()) {
            return;
          }
          try {
            extraReaders[w] = std::make_unique<ScopedCziReader>(filepath);
          } catch (const std::exception& e) {
            recordError(std::string("ScopedCziReader open failed: ") + e.what());
            return;
          } catch (...) {
            recordError("ScopedCziReader open failed");
            return;
          }
          openedExtra.fetch_add(1, std::memory_order_relaxed);
          runWorker(extraReaders[w]->reader());
        });
      }

      // Main thread acts as worker 0 using the cached reader, in parallel
      // with the extra-reader opens.
      runWorker(cziReader);
      for (auto& t : threads) {
        t.join();
      }
      auto tOpenEnd = std::chrono::high_resolution_clock::now();
      LOG_DEBUG << "CZI parallel HTTP load with " << numWorkers << " workers (" << jobs.size() << " planes); "
                << openedExtra.load() << " of " << numExtra
                << " extra readers opened in time to help; total open+read overlap "
                << std::chrono::duration<double, std::milli>(tOpenEnd - tOpenStart).count() << "ms";

      if (failed.load()) {
        LOG_ERROR << "Failed to read CZI plane(s) from " << filepath << ": " << errorMessage;
        return emptyimage;
      }
    }
    auto tReadEnd = std::chrono::high_resolution_clock::now();
    LOG_DEBUG << "CZI plane reads: " << std::chrono::duration<double, std::milli>(tReadEnd - tReadStart).count()
              << "ms for " << jobs.size() << " planes";

    auto tEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = tEnd - tStart;
    LOG_DEBUG << "CZI loaded in " << (elapsed.count() * 1000.0) << "ms";

    auto tStartImage = std::chrono::high_resolution_clock::now();

    // TODO: convert data to uint16_t pixels if not already.
    // we can release the smartPtr because ImageXYZC will now own the raw data memory
    ImageXYZC* im = new ImageXYZC(dims.sizeX,
                                  dims.sizeY,
                                  dims.sizeZ,
                                  nch,
                                  ImageXYZC::IN_MEMORY_BPP, // dims.bitsPerPixel,
                                  smartPtr.release(),
                                  dims.physicalSizeX,
                                  dims.physicalSizeY,
                                  dims.physicalSizeZ,
                                  dims.spatialUnits);

    std::vector<std::string> channelNames = dims.getChannelNames(loadSpec.channels);
    im->setChannelNames(channelNames);

    tEnd = std::chrono::high_resolution_clock::now();
    elapsed = tEnd - tStartImage;
    LOG_DEBUG << "ImageXYZC prepared in " << (elapsed.count() * 1000.0) << "ms";

    elapsed = tEnd - tStart;
    LOG_DEBUG << "Loaded " << filepath << " in " << (elapsed.count() * 1000.0) << "ms";

    std::shared_ptr<ImageXYZC> sharedImage(im);
    outDims = dims;

    return sharedImage;

  } catch (std::exception& e) {
    LOG_ERROR << e.what();
    LOG_ERROR << "Failed to read " << filepath;
    return emptyimage;
  } catch (...) {
    LOG_ERROR << "Failed to read " << filepath;
    return emptyimage;
  }
}

std::vector<MultiscaleDims>
FileReaderCzi::loadMultiscaleDims(const std::string& filepath, uint32_t scene)
{
  std::vector<MultiscaleDims> dims;
  VolumeDimensions vdims = loadDimensions(filepath, scene);
  if (!vdims.validate()) {
    return dims;
  }
  MultiscaleDims mdims;
  mdims.shape = { vdims.sizeT, vdims.sizeC, vdims.sizeZ, vdims.sizeY, vdims.sizeX };
  mdims.scale = { 1.0, 1.0, vdims.physicalSizeZ, vdims.physicalSizeY, vdims.physicalSizeX };
  mdims.dimensionOrder = { "T", "C", "Z", "Y", "X" };
  mdims.dtype = "uint16";
  mdims.path = "";
  mdims.channelNames = vdims.channelNames;
  dims.push_back(mdims);
  return dims;
}
