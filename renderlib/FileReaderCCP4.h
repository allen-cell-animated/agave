#pragma once

#include "VolumeDimensions.h"

#include <memory>
#include <string>

class ImageXYZC;

class FileReaderCCP4
{
public:
  FileReaderCCP4();
  virtual ~FileReaderCCP4();

  static std::shared_ptr<ImageXYZC> loadCCP4(const std::string& filepath,
                                             VolumeDimensions* dims = nullptr,
                                             uint32_t time = 0,
                                             uint32_t scene = 0);
  static VolumeDimensions loadDimensionsCCP4(const std::string& filepath, uint32_t scene = 0);
  static uint32_t loadNumScenesCCP4(const std::string& filepath);
};
