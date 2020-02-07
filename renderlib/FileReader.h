#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <QString>

class CBoundingBox;
class ImageXYZC;

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
};

class FileReader
{
public:
  FileReader();
  virtual ~FileReader();

  static std::shared_ptr<ImageXYZC> loadOMETiff_4D(const std::string& filepath, bool addToCache = false);
  static std::shared_ptr<ImageXYZC> loadFromArray_4D(uint8_t* dataArray,
                                                     std::vector<uint32_t> shape,
                                                     const std::string& name,
                                                     std::vector<char> dims = {},
                                                     std::vector<std::string> channelNames = {},
                                                     std::vector<float> physicalSizes = { 1.0f, 1.0f, 1.0f },
                                                     bool addToCache = false);
  // static void tiffReadChannelXYZ(uint8_t* byteptr, const VolumeDimensions& dims, int c, int t = 0);

private:
  static std::map<std::string, std::shared_ptr<ImageXYZC>> sPreloadedImageCache;
};
