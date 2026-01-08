#include "MultiFileCziStream.h"
#include "CziMultiFileUtils.h"
#include <filesystem>
#include <stdexcept>
#include <sstream>

namespace czi_multi {

MultiFileCziStream::MultiFileCziStream(const std::string& parentFilepath, const CziDirectoryInfo& dirInfo)
  : parentFilepath_(parentFilepath)
  , dirInfo_(dirInfo)
{
  // Pre-load the parent stream (file part 0)
  std::filesystem::path fpath(parentFilepath);
  auto parentStream = libCZI::CreateStreamFromFile(fpath.wstring().c_str());
  fileStreams_[0] = parentStream;
}

std::shared_ptr<libCZI::IStream>
MultiFileCziStream::LoadChildStream(std::int32_t filePart)
{
  if (filePart == 0) {
    return fileStreams_[0]; // Parent is always loaded
  }

  // Generate possible child file paths
  auto candidates = CziMultiFileUtils::GenerateChildFilePaths(parentFilepath_, filePart);

  for (const auto& candidate : candidates) {
    if (std::filesystem::exists(candidate)) {
      // Validate GUID
      if (CziMultiFileUtils::ValidateFileGuid(candidate, dirInfo_.primaryFileGuid)) {
        std::filesystem::path fpath(candidate);
        auto stream = libCZI::CreateStreamFromFile(fpath.wstring().c_str());
        return stream;
      }
    }
  }

  // Child file not found
  return nullptr;
}

std::shared_ptr<libCZI::IStream>
MultiFileCziStream::GetStreamForFilePart(std::int32_t filePart)
{
  std::lock_guard<std::mutex> lock(streamMutex_);

  auto it = fileStreams_.find(filePart);
  if (it != fileStreams_.end()) {
    return it->second;
  }

  // Lazy load
  auto stream = LoadChildStream(filePart);
  if (stream) {
    fileStreams_[filePart] = stream;
  }

  return stream;
}

std::int32_t
MultiFileCziStream::FindFilePartForPosition(std::uint64_t position) const
{
  auto it = dirInfo_.filePositionToFilePart.find(position);
  if (it != dirInfo_.filePositionToFilePart.end()) {
    return it->second;
  }

  // If not found in map, assume it's in the parent file
  return 0;
}

void
MultiFileCziStream::Read(std::uint64_t offset, void* pv, std::uint64_t size, std::uint64_t* ptrBytesRead)
{
  // Find which file part contains this offset
  std::int32_t filePart = FindFilePartForPosition(offset);

  // Get the appropriate stream
  auto stream = GetStreamForFilePart(filePart);

  if (!stream) {
    std::ostringstream ss;
    ss << "Failed to open child file for FilePart=" << filePart << " (offset " << offset << ")";
    throw std::runtime_error(ss.str());
  }

  // Delegate the read to the appropriate stream
  stream->Read(offset, pv, size, ptrBytesRead);
}

bool
MultiFileCziStream::ValidateChildFiles()
{
  for (std::int32_t filePart : dirInfo_.uniqueFileParts) {
    if (filePart == 0)
      continue; // Parent is always available

    auto stream = GetStreamForFilePart(filePart);
    if (!stream) {
      return false;
    }
  }

  return true;
}

std::vector<std::int32_t>
MultiFileCziStream::GetMissingFileParts()
{
  std::vector<std::int32_t> missing;

  for (std::int32_t filePart : dirInfo_.uniqueFileParts) {
    if (filePart == 0)
      continue; // Parent is always available

    auto stream = GetStreamForFilePart(filePart);
    if (!stream) {
      missing.push_back(filePart);
    }
  }

  return missing;
}

} // namespace czi_multi
