#pragma once

#include "version.h"

#include <array>
class Version
{
public:
  uint32_t major() const { return mVersion[0]; }
  uint32_t minor() const { return mVersion[1]; }
  uint32_t patch() const { return mVersion[2]; }
  std::array<uint32_t, 3> mVersion;

  Version(const std::array<uint32_t, 3>& version)
    : mVersion(version)
  {
  }
  Version(uint32_t major, uint32_t minor, uint32_t patch)
    : mVersion{ major, minor, patch }
  {
  }

  bool operator==(const Version& v) const { return compare(v) == 0; };
  bool operator!=(const Version& v) const { return !(operator==(v)); };

  bool operator<(const Version& v) const { return compare(v) < 0; };
  bool operator>(const Version& v) const { return v.operator<(*this); };

  bool operator<=(const Version& v) const { return !(operator>(v)); };
  bool operator>=(const Version& v) const { return !(operator<(v)); };

private:
  int compare(const Version& v) const
  {
    if (major() > v.major()) {
      return 1;
    } else if (major() < v.major()) {
      return -1;
    } else {
      if (minor() > v.minor()) {
        return 1;
      } else if (minor() < v.minor()) {
        return -1;
      } else {
        if (patch() > v.patch()) {
          return 1;
        } else if (patch() < v.patch()) {
          return -1;
        } else {
          return 0;
        }
      }
    }
  }
};
