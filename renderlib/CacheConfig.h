#pragma once

#include <cstdint>
#include <string>

namespace renderlib {

struct CacheConfig
{
  bool enabled = false;
  bool enableDisk = false;
  std::uint64_t maxRamBytes = 0;
  std::uint64_t maxDiskBytes = 0;
  std::string cacheDir;
};

} // namespace renderlib
