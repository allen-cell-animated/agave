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

  static std::shared_ptr<ImageXYZC> loadFromFile_4D(const std::string& filepath, bool addToCache = false);
  static std::shared_ptr<ImageXYZC> loadFromArray_4D(uint8_t* dataArray,
                                                     std::vector<uint32_t> shape,
                                                     const std::string& name,
                                                     std::vector<char> dims = {},
                                                     std::vector<std::string> channelNames = {},
                                                     std::vector<float> physicalSizes = { 1.0f, 1.0f, 1.0f },
                                                     bool addToCache = false);

private:
  static std::map<std::string, std::shared_ptr<ImageXYZC>> sPreloadedImageCache;
};
