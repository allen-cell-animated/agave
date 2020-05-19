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

  static std::shared_ptr<ImageXYZC> loadCzi(const std::string& filepath,
                                            VolumeDimensions* dims = nullptr,
                                            uint32_t time = 0,
                                            uint32_t scene = 0);
  static VolumeDimensions loadDimensionsCzi(const std::string& filepath, uint32_t scene = 0);
  static uint32_t loadNumScenesCzi(const std::string& filepath);
};
