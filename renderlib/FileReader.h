#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include <QString>

class CBoundingBox;
class ImageXYZC;

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

private:
  static std::map<std::string, std::shared_ptr<ImageXYZC>> sPreloadedImageCache;

  static void getZCT(uint32_t i,
                     QString dimensionOrder,
                     uint32_t size_z,
                     uint32_t size_c,
                     uint32_t size_t,
                     uint32_t& z,
                     uint32_t& c,
                     uint32_t& t);
};
