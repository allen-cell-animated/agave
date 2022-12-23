#pragma once

#include "VolumeDimensions.h"

#include <memory>
#include <string>

class ImageXYZC;

class FileReaderZarr
{
public:
  FileReaderZarr();
  virtual ~FileReaderZarr();

  static std::shared_ptr<ImageXYZC> loadOMEZarr(const std::string& filepath,
                                                VolumeDimensions* dims = nullptr,
                                                uint32_t time = 0,
                                                uint32_t scene = 0);
  static VolumeDimensions loadDimensionsZarr(const std::string& filepath, uint32_t scene = 0);
  static uint32_t loadNumScenesZarr(const std::string& filepath);
  static std::vector<MultiscaleDims> loadMultiscaleDims(const std::string& filepath, uint32_t scene = 0);
};
