#pragma once

#include "IFileReader.h"
#include "VolumeDimensions.h"

#include <memory>
#include <string>

class ImageXYZC;

class FileReaderTIFF : public IFileReader
{
public:
  FileReaderTIFF(const std::string& filepath);
  virtual ~FileReaderTIFF();

  bool supportChunkedLoading() const { return false; }

  std::shared_ptr<ImageXYZC> loadFromFile(const LoadSpec& loadSpec);
  VolumeDimensions loadDimensions(const std::string& filepath, uint32_t scene = 0);
  uint32_t loadNumScenes(const std::string& filepath);
  std::vector<MultiscaleDims> loadMultiscaleDims(const std::string& filepath, uint32_t scene = 0);
};
