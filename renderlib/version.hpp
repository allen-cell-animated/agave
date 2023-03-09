#pragma once

#include "version.h"

#include <glm/glm.hpp>

class Version
{
public:
  int major() const { return mVersion[0]; }
  int minor() const { return mVersion[1]; }
  int patch() const { return mVersion[2]; }
  glm::ivec3 mVersion;

  Version(const glm::ivec3& version)
    : mVersion(version)
  {}
  Version(int major, int minor, int patch)
    : mVersion(major, minor, patch)
  {}

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

extern const Version CURRENT_VERSION(AICS_VERSION_MAJOR, AICS_VERSION_MINOR, AICS_VERSION_PATCH);