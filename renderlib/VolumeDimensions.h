#pragma once

#include <string>
#include <vector>

struct VolumeDimensions
{
  uint32_t sizeX = 1;
  uint32_t sizeY = 1;
  uint32_t sizeZ = 1;
  uint32_t sizeC = 1;
  uint32_t sizeT = 1;
  float physicalSizeX = 1.0f;
  float physicalSizeY = 1.0f;
  float physicalSizeZ = 1.0f;
  uint32_t bitsPerPixel = 16;
  std::string dimensionOrder = "XYZCT";
  std::vector<std::string> channelNames;

  uint32_t getPlaneIndex(uint32_t z, uint32_t c, uint32_t t) const;
  std::vector<uint32_t> getPlaneZCT(uint32_t planeIndex) const;

  bool validate() const;
  void log() const;
};
