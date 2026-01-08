#pragma once

#include "CziStructures.h"
#include "libCZI/Src/libCZI/libCZI.h"
#include <memory>
#include <map>
#include <set>
#include <string>

namespace czi_multi {

// Result of parsing the CZI directory
struct CziDirectoryInfo
{
  libCZI::GUID primaryFileGuid;
  libCZI::GUID fileGuid;
  std::int32_t filePart;
  int entryCount;

  // Map: filePosition -> filePart
  // This tells us which child file contains data at each file position
  std::map<std::uint64_t, std::int32_t> filePositionToFilePart;

  // Set of all unique file parts referenced
  std::set<std::int32_t> uniqueFileParts;

  bool IsMultiFile() const
  {
    return uniqueFileParts.size() > 1 || (uniqueFileParts.size() == 1 && *uniqueFileParts.begin() != 0);
  }

  bool IsChildFile() const { return filePart > 0; }

  int GetMaxFilePart() const
  {
    if (uniqueFileParts.empty())
      return 0;
    return *uniqueFileParts.rbegin();
  }
};

// Parser for CZI directory structure
class CziDirectoryParser
{
public:
  // Parse the directory from a CZI file stream
  // This reads the file header and subblock directory to extract FilePart information
  static CziDirectoryInfo ParseDirectory(libCZI::IStream* stream);

  // Parse just the file header to check if it's a child file
  static FileHeaderSegment ReadFileHeader(libCZI::IStream* stream);

  // Check if two GUIDs are equal
  static bool GuidsEqual(const libCZI::GUID& a, const libCZI::GUID& b);

  // Format GUID as string for debugging
  static std::string GuidToString(const libCZI::GUID& guid);

private:
  static SubBlockDirectorySegment ReadDirectorySegment(libCZI::IStream* stream, std::uint64_t position);
  static void ParseDirectoryEntries(libCZI::IStream* stream,
                                    std::uint64_t basePosition,
                                    std::int32_t entryCount,
                                    CziDirectoryInfo& info);
  static std::uint64_t ParseDVEntry(libCZI::IStream* stream, std::uint64_t position, CziDirectoryInfo& info);
  static std::uint64_t ParseDEEntry(libCZI::IStream* stream, std::uint64_t position, CziDirectoryInfo& info);
};

} // namespace czi_multi
