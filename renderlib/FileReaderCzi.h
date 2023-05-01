#pragma once

#include "IFileReader.h"
#include "VolumeDimensions.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

class CBoundingBox;
class ImageXYZC;

class FileReaderCzi : public IFileReader
{
public:
  FileReaderCzi(const std::string& filepath);
  virtual ~FileReaderCzi();

  bool supportChunkedLoading() const { return false; }

  std::shared_ptr<ImageXYZC> loadFromFile(const LoadSpec& loadSpec);
  VolumeDimensions loadDimensions(const std::string& filepath, uint32_t scene = 0);
  uint32_t loadNumScenes(const std::string& filepath);
  std::vector<MultiscaleDims> loadMultiscaleDims(const std::string& filepath, uint32_t scene = 0);
};
