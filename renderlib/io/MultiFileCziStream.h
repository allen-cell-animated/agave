#pragma once

#include "CziDirectoryParser.h"
#include "libCZI/Src/libCZI/libCZI.h"
#include <memory>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace czi_multi {

// Multi-file stream implementation that routes reads to appropriate child files
// based on file position mapping from the CZI directory
class MultiFileCziStream : public libCZI::IStream
{
public:
  // Constructor takes the parent file path and directory info
  MultiFileCziStream(const std::string& parentFilepath, const CziDirectoryInfo& dirInfo);

  // IStream interface
  void Read(std::uint64_t offset, void* pv, std::uint64_t size, std::uint64_t* ptrBytesRead) override;

  // Get the underlying stream for a specific file part
  // This is used by libCZI when reading subblocks
  std::shared_ptr<libCZI::IStream> GetStreamForFilePart(std::int32_t filePart);

  // Check if all required child files are accessible
  bool ValidateChildFiles();

  // Get list of missing child files (if any)
  std::vector<std::int32_t> GetMissingFileParts();

private:
  std::string parentFilepath_;
  CziDirectoryInfo dirInfo_;

  // Map of file part -> stream
  std::map<std::int32_t, std::shared_ptr<libCZI::IStream>> fileStreams_;

  // Mutex for thread-safe lazy loading of streams
  mutable std::mutex streamMutex_;

  // Lazy-load a child file stream
  std::shared_ptr<libCZI::IStream> LoadChildStream(std::int32_t filePart);

  // Find the file part for a given file position
  std::int32_t FindFilePartForPosition(std::uint64_t position) const;
};

} // namespace czi_multi
