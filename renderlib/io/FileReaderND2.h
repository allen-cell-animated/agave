#pragma once

#include "IFileReader.h"
#include "VolumeDimensions.h"

#include <memory>
#include <string>

class ImageXYZC;

// Reader for Nikon ND2 image files (v3, post-2017 chunkmap format).
//
// Implementation notes:
// - Only local-file access is supported for now (no HTTP/S3).
// - Multiscene (XY positions) and time-series loading are supported, mirroring
//   FileReaderCzi / FileReaderZarr semantics: loadDimensions / loadFromFile
//   take a (scene, time) and return one volume.
// - The format details (chunkmap footer, CLX-Lite metadata variant, frame
//   chunk decompression) are implemented natively from the published nd2
//   format spec adapted from tlambert03/nd2 (BSD-3 reference). zlib and lz4
//   compressed frames are both supported.
class FileReaderND2 : public IFileReader
{
public:
  FileReaderND2(const std::string& filepath);
  ~FileReaderND2() override;

  bool supportChunkedLoading() const override { return false; }

  std::shared_ptr<ImageXYZC> loadFromFile(const LoadSpec& loadSpec) override;
  VolumeDimensions loadDimensions(const std::string& filepath, uint32_t scene = 0) override;
  uint32_t loadNumScenes(const std::string& filepath) override;
  std::vector<MultiscaleDims> loadMultiscaleDims(const std::string& filepath, uint32_t scene = 0) override;
};
