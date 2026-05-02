#include <catch2/catch_test_macros.hpp>

#include "renderlib/CacheConfig.h"
#include "renderlib/CacheManager.h"
#include "renderlib/IFileReader.h"
#include "renderlib/ImageXYZC.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

namespace {

// Bytes per pixel that CacheManager uses when estimating an image's RAM cost.
// IN_MEMORY_BPP is a static const member without an out-of-class definition,
// so we copy its value into a constexpr to avoid ODR-use at the link step.
constexpr std::uint32_t kInMemoryBpp = 16;
constexpr std::uint64_t kBytesPerPixel = kInMemoryBpp / 8;
static_assert(kInMemoryBpp == ImageXYZC::IN_MEMORY_BPP, "IN_MEMORY_BPP changed");

// Build a minimal in-memory ImageXYZC for cache testing. Memory is owned by the
// returned ImageXYZC (it deletes m_data in its destructor).
std::shared_ptr<ImageXYZC>
makeImage(uint32_t x, uint32_t y, uint32_t z, uint32_t c)
{
  const std::uint64_t bytes =
    static_cast<std::uint64_t>(x) * y * z * c * kBytesPerPixel;
  auto* data = new uint8_t[bytes];
  std::memset(data, 0, bytes);
  return std::make_shared<ImageXYZC>(x, y, z, c, kInMemoryBpp, data, 1.0f, 1.0f, 1.0f, "units");
}

LoadSpec
makeSpec(const std::string& filepath, uint32_t time = 0)
{
  LoadSpec s;
  s.filepath = filepath;
  s.time = time;
  return s;
}

std::uint64_t
imageBytes(uint32_t x, uint32_t y, uint32_t z, uint32_t c)
{
  return static_cast<std::uint64_t>(x) * y * z * c * kBytesPerPixel;
}

// Reset the singleton CacheManager to a known state for each test.
void
resetCache()
{
  CacheManager::instance().clear();
  CacheManager::instance().resetStats();
}

CacheConfig
ramOnlyConfig(std::uint64_t maxRamBytes)
{
  CacheConfig cfg;
  cfg.enabled = true;
  cfg.enableDisk = false;
  cfg.maxRamBytes = maxRamBytes;
  cfg.maxDiskBytes = 0;
  return cfg;
}

} // namespace

TEST_CASE("CacheManager respects RAM limit and evicts LRU entries", "[cache]")
{
  // Each 4x4x4x1 image is 4*4*4*1*2 = 128 bytes.
  const std::uint64_t oneImage = imageBytes(4, 4, 4, 1);

  SECTION("Store and retrieve a single image (RAM hit)")
  {
    resetCache();
    CacheManager::instance().setConfig(ramOnlyConfig(oneImage * 4));

    auto img = makeImage(4, 4, 4, 1);
    auto spec = makeSpec("a");

    CacheManager::instance().storeImage(spec, img);
    auto retrieved = CacheManager::instance().findImage(spec);

    REQUIRE(retrieved != nullptr);
    REQUIRE(retrieved.get() == img.get());

    auto stats = CacheManager::instance().getStats();
    REQUIRE(stats.ramHits == 1);
    REQUIRE(stats.misses == 0);
  }

  SECTION("Miss returns null and increments miss counter")
  {
    resetCache();
    CacheManager::instance().setConfig(ramOnlyConfig(oneImage * 4));

    auto retrieved = CacheManager::instance().findImage(makeSpec("missing"));
    REQUIRE(retrieved == nullptr);
    REQUIRE(CacheManager::instance().getStats().misses == 1);
  }

  SECTION("Cache stays under maxRamBytes when limit is reached")
  {
    resetCache();
    // Limit is exactly 2 images. Storing a 3rd must evict.
    CacheManager::instance().setConfig(ramOnlyConfig(oneImage * 2));

    CacheManager::instance().storeImage(makeSpec("a"), makeImage(4, 4, 4, 1));
    CacheManager::instance().storeImage(makeSpec("b"), makeImage(4, 4, 4, 1));
    CacheManager::instance().storeImage(makeSpec("c"), makeImage(4, 4, 4, 1));

    // "a" was the least recently used and must be evicted.
    REQUIRE(CacheManager::instance().findImage(makeSpec("a")) == nullptr);
    REQUIRE(CacheManager::instance().findImage(makeSpec("b")) != nullptr);
    REQUIRE(CacheManager::instance().findImage(makeSpec("c")) != nullptr);
  }

  SECTION("LRU ordering is updated on access (touched entry survives eviction)")
  {
    resetCache();
    CacheManager::instance().setConfig(ramOnlyConfig(oneImage * 2));

    CacheManager::instance().storeImage(makeSpec("a"), makeImage(4, 4, 4, 1));
    CacheManager::instance().storeImage(makeSpec("b"), makeImage(4, 4, 4, 1));

    // Touch "a" so that "b" becomes least recently used.
    REQUIRE(CacheManager::instance().findImage(makeSpec("a")) != nullptr);

    CacheManager::instance().storeImage(makeSpec("c"), makeImage(4, 4, 4, 1));

    REQUIRE(CacheManager::instance().findImage(makeSpec("a")) != nullptr);
    REQUIRE(CacheManager::instance().findImage(makeSpec("b")) == nullptr);
    REQUIRE(CacheManager::instance().findImage(makeSpec("c")) != nullptr);
  }

  SECTION("Storing many images keeps total RAM usage bounded")
  {
    resetCache();
    const std::uint64_t cap = oneImage * 3;
    CacheManager::instance().setConfig(ramOnlyConfig(cap));

    for (int i = 0; i < 20; ++i) {
      CacheManager::instance().storeImage(makeSpec("file_" + std::to_string(i)), makeImage(4, 4, 4, 1));
    }

    // Count how many of the 20 inserts are still resident. With a cap of 3
    // images, no more than 3 can be present at once.
    CacheManager::instance().resetStats();
    int present = 0;
    for (int i = 0; i < 20; ++i) {
      if (CacheManager::instance().findImage(makeSpec("file_" + std::to_string(i)))) {
        present++;
      }
    }
    REQUIRE(present <= 3);
    // The most recently inserted entries should still be in the cache.
    REQUIRE(present >= 1);
    REQUIRE(CacheManager::instance().findImage(makeSpec("file_19")) != nullptr);
  }

  SECTION("Image larger than maxRamBytes is not stored")
  {
    resetCache();
    // Cap at less than the size of one image.
    CacheManager::instance().setConfig(ramOnlyConfig(oneImage / 2));

    CacheManager::instance().storeImage(makeSpec("too_big"), makeImage(4, 4, 4, 1));
    REQUIRE(CacheManager::instance().findImage(makeSpec("too_big")) == nullptr);
  }

  SECTION("Re-storing the same key replaces the entry without exceeding the limit")
  {
    resetCache();
    CacheManager::instance().setConfig(ramOnlyConfig(oneImage * 2));

    auto first = makeImage(4, 4, 4, 1);
    auto second = makeImage(4, 4, 4, 1);
    auto spec = makeSpec("a");

    CacheManager::instance().storeImage(spec, first);
    CacheManager::instance().storeImage(spec, second);

    // Fill the cache up to the limit with another entry. If the duplicate
    // key had double-counted bytes, this would evict "a".
    CacheManager::instance().storeImage(makeSpec("b"), makeImage(4, 4, 4, 1));

    auto retrieved = CacheManager::instance().findImage(spec);
    REQUIRE(retrieved != nullptr);
    REQUIRE(retrieved.get() == second.get());
    REQUIRE(CacheManager::instance().findImage(makeSpec("b")) != nullptr);
  }

  SECTION("Disabling the cache clears all entries")
  {
    resetCache();
    CacheManager::instance().setConfig(ramOnlyConfig(oneImage * 4));
    CacheManager::instance().storeImage(makeSpec("a"), makeImage(4, 4, 4, 1));
    REQUIRE(CacheManager::instance().findImage(makeSpec("a")) != nullptr);

    CacheConfig disabled;
    disabled.enabled = false;
    CacheManager::instance().setConfig(disabled);

    REQUIRE(CacheManager::instance().findImage(makeSpec("a")) == nullptr);
  }

  SECTION("Shrinking the limit evicts existing entries")
  {
    resetCache();
    CacheManager::instance().setConfig(ramOnlyConfig(oneImage * 4));
    CacheManager::instance().storeImage(makeSpec("a"), makeImage(4, 4, 4, 1));
    CacheManager::instance().storeImage(makeSpec("b"), makeImage(4, 4, 4, 1));
    CacheManager::instance().storeImage(makeSpec("c"), makeImage(4, 4, 4, 1));

    // Reconfigure to a smaller limit; oldest entries must be evicted.
    CacheManager::instance().setConfig(ramOnlyConfig(oneImage));

    int present = 0;
    for (const char* name : { "a", "b", "c" }) {
      if (CacheManager::instance().findImage(makeSpec(name))) {
        present++;
      }
    }
    REQUIRE(present <= 1);
  }

  SECTION("clear() empties the cache")
  {
    resetCache();
    CacheManager::instance().setConfig(ramOnlyConfig(oneImage * 4));
    CacheManager::instance().storeImage(makeSpec("a"), makeImage(4, 4, 4, 1));
    CacheManager::instance().storeImage(makeSpec("b"), makeImage(4, 4, 4, 1));

    CacheManager::instance().clear();

    REQUIRE(CacheManager::instance().findImage(makeSpec("a")) == nullptr);
    REQUIRE(CacheManager::instance().findImage(makeSpec("b")) == nullptr);
  }
}
