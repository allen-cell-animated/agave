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
  // last_write_time of the filepath (or directory) at the time the key was
  // built, expressed as nanoseconds since epoch. Zero for remote URLs and for
  // paths we couldn't stat. Folding this into the key invalidates cache
  // entries when the source file is overwritten.
  std::uint64_t fileMtimeNs = 0;
  // file_size of filepath at the time the key was built. Zero for
  // directories (zarr) and remote URLs.
  std::uint64_t fileSize = 0;

  bool operator==(const CacheKey& other) const;
};

struct CacheKeyHash
{
  std::size_t operator()(const CacheKey& key) const;
};

class CacheManager
{
  // Test-only access to internals. Lets the test suite re-point the cache
  // directory between cases (the singleton is shared across all tests) without
  // going through the once-only initialize() guard. Declared here so the
  // production API exposes no test hooks.
  friend struct CacheManagerTestAccess;

public:
  struct CacheStats
  {
    std::uint64_t ramHits = 0;
    std::uint64_t diskHits = 0;
    std::uint64_t misses = 0;
    std::uint64_t diskWrites = 0;
  };

  static CacheManager& instance();

  // Register the on-disk cache root. Must be called exactly once, at app
  // startup, before the cache is used. The platform-appropriate path is
  // resolved by the GUI layer (renderlib has no Qt and so cannot resolve
  // QStandardPaths itself) and injected here. The cache directory is fixed for
  // the lifetime of the process and is never derived from per-apply
  // CacheConfig. Calling initialize() more than once is a programming error
  // and throws std::logic_error.
  static void initialize(const std::string& cacheDir);
  std::string getCacheDirectory() const;

  void setConfig(const CacheConfig& config);
  CacheConfig getConfig() const;

  std::shared_ptr<ImageXYZC> findImage(const LoadSpec& loadSpec);
  void storeImage(const LoadSpec& loadSpec, const std::shared_ptr<ImageXYZC>& image);
  // Drop all entries from the in-memory cache. Disk cache is untouched.
  void clearMemoryCache();
  // Drop all entries from the disk cache (refuses if the cache directory is
  // missing the AGAVE marker file). Memory cache is untouched.
  void clearDiskCache();

  CacheStats getStats() const;
  void resetStats();

private:
  CacheManager() = default;

  // Verify that `path` is (or can be made) a writable directory. Creates the
  // directory if it does not exist, then probes it by writing and deleting a
  // small marker file. Returns false on any failure. Probed once, at
  // initialize() time, since the cache root is fixed for the process lifetime.
  static bool canWriteCacheDir(const std::string& path);

  // Records the cache root and (re)builds the disk index for it when the disk
  // tier is active. The single production entry point is initialize(); this
  // worker is also reused by the test seam to re-point across test cases.
  void applyCacheDirectory(const std::string& dir);

  CacheKey makeKey(const LoadSpec& loadSpec) const;
  std::string keyToString(const CacheKey& key) const;
  std::string diskCacheId(const CacheKey& key) const;
  std::uint64_t estimateImageBytes(const ImageXYZC& image) const;
  void touchEntry(std::list<CacheKey>::iterator it);
  // Precondition: caller must hold m_mutex.
  void evictIfNeededLocked(std::uint64_t incomingBytes);
  void storeImageInMemory(const CacheKey& key, const std::shared_ptr<ImageXYZC>& image);

  std::shared_ptr<ImageXYZC> loadFromDisk(const CacheKey& key, const CacheConfig& config, const std::string& cacheDir);
  void storeToDisk(const CacheKey& key,
                   const std::shared_ptr<ImageXYZC>& image,
                   const CacheConfig& config,
                   const std::string& cacheDir);
  void loadDiskIndex(const CacheConfig& config, const std::string& cacheDir);
  void evictDiskIfNeeded(const CacheConfig& config, std::uint64_t incomingBytes);
  std::uint64_t directorySizeBytes(const std::string& path) const;
  // Writes a marker file to a directory we manage as our own disk cache root.
  // clearDiskCache refuses to delete anything unless this marker is present,
  // protecting against accidental wipes of user-typed paths (e.g. "C:\").
  void writeCacheMarker(const std::string& path) const;
  bool isAgaveCacheDir(const std::string& path) const;

  mutable std::mutex m_mutex;
  CacheConfig m_config;
  // Set true by initialize(); guards against a second initialize() call.
  bool m_initialized = false;
  // The disk cache root, registered once via initialize(). Distinct from
  // m_diskIndexRoot, which tracks the root the in-memory index was last built
  // against (used to decide when a rebuild is needed).
  std::string m_cacheDir;
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
