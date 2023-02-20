#pragma once

#include "IFileReader.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

class ImageXYZC;
struct VolumeDimensions;
struct MultiscaleDims;

class FileReader
{
public:
  FileReader();
  virtual ~FileReader();

  static IFileReader* getReader(const std::string& filepath);

  static uint32_t loadNumScenes(const std::string& filepath);

  // return dimensions from scene in file
  static VolumeDimensions loadFileDimensions(const std::string& filepath, uint32_t scene = 0);
  static bool loadMultiscaleDims(const std::string& filepath, uint32_t scene, std::vector<MultiscaleDims>& dims);

  static std::shared_ptr<ImageXYZC> loadFromFile(const LoadSpec& loadSpec, bool addToCache = false);

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
