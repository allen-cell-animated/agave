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
  ~FileReaderTIFF() override;

  bool supportChunkedLoading() const override { return false; }

  std::shared_ptr<ImageXYZC> loadFromFile(const LoadSpec& loadSpec) override;
  VolumeDimensions loadDimensions(const std::string& filepath, uint32_t scene = 0) override;
  uint32_t loadNumScenes(const std::string& filepath) override;
  std::vector<MultiscaleDims> loadMultiscaleDims(const std::string& filepath, uint32_t scene = 0) override;
};
