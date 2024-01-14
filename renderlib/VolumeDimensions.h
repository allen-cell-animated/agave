#pragma once

#include <cstdint>
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
  // currently assumes xyz same unit
  std::string spatialUnits = "units";
  uint32_t bitsPerPixel = 16;
  // SAMPLEFORMAT_UINT = 1;
  // SAMPLEFORMAT_INT = 2;
  // SAMPLEFORMAT_IEEEFP = 3;
  uint16_t sampleFormat = 1;
  std::string dimensionOrder = "XYZCT";
  std::vector<std::string> channelNames;

  uint32_t getPlaneIndex(uint32_t z, uint32_t c, uint32_t t) const;
  std::vector<uint32_t> getPlaneZCT(uint32_t planeIndex) const;

  bool validate() const;
  void log() const;

  std::vector<std::string> getChannelNames(const std::vector<uint32_t>& channels) const;

  static std::string sanitizeUnitsString(std::string units);
};

struct MultiscaleDims
{
  std::vector<float> scale;
  std::vector<int64_t> shape;
  std::vector<std::string> dimensionOrder;
  std::string dtype;
  std::string path;
  std::vector<std::string> channelNames;
  std::string spatialUnits = "units";

  bool hasDim(const std::string& dim) const;
  int64_t sizeT() const;
  int64_t sizeC() const;
  int64_t sizeZ() const;
  int64_t sizeY() const;
  int64_t sizeX() const;
  float scaleX() const;
  float scaleY() const;
  float scaleZ() const;

  VolumeDimensions getVolumeDimensions() const;
};
