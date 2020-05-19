#pragma once

#include "VolumeDimensions.h"

#include <memory>
#include <string>

class ImageXYZC;

class FileReaderTIFF
{
public:
  FileReaderTIFF();
  virtual ~FileReaderTIFF();

  static std::shared_ptr<ImageXYZC> loadOMETiff(const std::string& filepath,
                                                VolumeDimensions* dims = nullptr,
                                                uint32_t time = 0,
                                                uint32_t scene = 0);
  static VolumeDimensions loadDimensionsTiff(const std::string& filepath, uint32_t scene = 0);
  static uint32_t loadNumScenesTiff(const std::string& filepath);
};
