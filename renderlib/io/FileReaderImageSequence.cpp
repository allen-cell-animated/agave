#include "FileReaderImageSequence.h"

#include "FileReaderTIFF.h"

#include <algorithm>
#include <filesystem>

std::vector<std::string>
initializeSequence(const std::string& filepath)
{
  std::filesystem::path fpath(filepath);
  // return a listing of all files in directory of filepath with same file extension
  std::vector<std::string> files;
  std::filesystem::path directory = fpath.parent_path();
  std::filesystem::path extension = fpath.extension();

  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    if (entry.is_regular_file() && entry.path().extension() == extension) {
      files.push_back(entry.path().string());
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

FileReaderImageSequence::FileReaderImageSequence(const std::string& filepath)
  : m_tiffReader(new FileReaderTIFF(filepath))
{
  m_sequence = initializeSequence(filepath);
}

FileReaderImageSequence::~FileReaderImageSequence() {}

std::shared_ptr<ImageXYZC>
FileReaderImageSequence::loadFromFile(const LoadSpec& loadSpec)
{
  LoadSpec sequenceSpec = loadSpec;
  sequenceSpec.filepath = m_sequence[loadSpec.time];
  sequenceSpec.time = 0;
  return m_tiffReader->loadFromFile(sequenceSpec);
}

VolumeDimensions
FileReaderImageSequence::loadDimensions(const std::string& filepath, uint32_t scene)
{
  VolumeDimensions vd = m_tiffReader->loadDimensions(filepath, scene);
  vd.sizeT = m_sequence.size();
  return vd;
}

uint32_t
FileReaderImageSequence::loadNumScenes(const std::string& filepath)
{
  return m_tiffReader->loadNumScenes(filepath);
}

std::vector<MultiscaleDims>
FileReaderImageSequence::loadMultiscaleDims(const std::string& filepath, uint32_t scene)
{
  std::vector<MultiscaleDims> dims = m_tiffReader->loadMultiscaleDims(filepath, scene);
  for (auto& d : dims) {
    d.shape[0] = m_sequence.size();
  }
  return dims;
}
