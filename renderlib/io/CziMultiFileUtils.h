#pragma once

#include "CziDirectoryParser.h"
#include <string>
#include <vector>
#include <optional>
#include <filesystem>

namespace czi_multi {

// Utilities for discovering and validating multi-file CZI sets
class CziMultiFileUtils
{
public:
  // Check if a file is part of a multi-file CZI set
  static bool IsMultiFileCzi(const std::string& filepath);

  // Check if a file is a child CZI file (FilePart > 0)
  static bool IsChildCziFile(const std::string& filepath);

  // Find the parent file given a child file path
  // Returns empty optional if parent cannot be found
  static std::optional<std::string> FindParentFile(const std::string& childFilepath);

  // Find all child files for a given parent file
  // Returns vector of child file paths (does not include parent)
  static std::vector<std::string> FindChildFiles(const std::string& parentFilepath);

  // Generate child file path for a given file part number
  // Tries multiple naming conventions
  static std::vector<std::string> GenerateChildFilePaths(const std::string& parentFilepath, int filePart);

  // Validate that a file has the expected primary GUID
  static bool ValidateFileGuid(const std::string& filepath, const libCZI::GUID& expectedPrimaryGuid);

  // Get file info without opening full CZI reader
  struct FileInfo
  {
    libCZI::GUID primaryFileGuid;
    libCZI::GUID fileGuid;
    std::int32_t filePart;
    bool isChildFile;
    bool isMultiFile;
  };

  static FileInfo GetFileInfo(const std::string& filepath);

private:
  // Extract parent name from child filename patterns
  static std::optional<std::string> ExtractParentNameFromPattern(const std::string& filename);
};

} // namespace czi_multi
