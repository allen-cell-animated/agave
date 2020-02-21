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
                                                int32_t time = 0,
                                                int32_t scene = 0);
  static VolumeDimensions loadDimensionsTiff(const std::string& filepath, int32_t scene = 0);
};
