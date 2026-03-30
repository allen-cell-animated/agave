#pragma once

#include "IFileReader.h"
#include "VolumeDimensions.h"

#include <memory>
#include <string>
#include <vector>

class ImageXYZC;
class FileReaderTIFF;

class FileReaderImageSequence : public IFileReader
{
public:
  FileReaderImageSequence(const std::string& filepath);
  ~FileReaderImageSequence() override;

  bool supportChunkedLoading() const override { return false; }

  std::shared_ptr<ImageXYZC> loadFromFile(const LoadSpec& loadSpec) override;
  VolumeDimensions loadDimensions(const std::string& filepath, uint32_t scene = 0) override;
  uint32_t loadNumScenes(const std::string& filepath) override;
  std::vector<MultiscaleDims> loadMultiscaleDims(const std::string& filepath, uint32_t scene = 0) override;

private:
  std::unique_ptr<FileReaderTIFF> m_tiffReader;
  std::vector<std::string> m_sequence;
};
