#pragma once

#include "VolumeDimensions.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

class CBoundingBox;
class ImageXYZC;

class FileReaderCzi
{
public:
  FileReaderCzi();
  virtual ~FileReaderCzi();

  static std::shared_ptr<ImageXYZC> loadCzi(const std::string& filepath, int32_t time = 0, int32_t scene = 0);
  static VolumeDimensions loadDimensionsCzi(const std::string& filepath, int32_t scene = 0);
};
