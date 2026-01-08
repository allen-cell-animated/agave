#include "CziMultiFileUtils.h"
#include "libCZI/Src/libCZI/libCZI.h"
#include <regex>
#include <iomanip>
#include <sstream>

namespace czi_multi {

CziMultiFileUtils::FileInfo
CziMultiFileUtils::GetFileInfo(const std::string& filepath)
{
  FileInfo info{};

  try {
    std::filesystem::path fpath(filepath);
    auto stream = libCZI::CreateStreamFromFile(fpath.wstring().c_str());

    auto header = CziDirectoryParser::ReadFileHeader(stream.get());

    info.primaryFileGuid = header.data.PrimaryFileGuid;
    info.fileGuid = header.data.FileGuid;
    info.filePart = header.data.FilePart;
    info.isChildFile = header.data.FilePart > 0;
    info.isMultiFile = !CziDirectoryParser::GuidsEqual(header.data.PrimaryFileGuid, header.data.FileGuid);

  } catch (...) {
    // If we can't read the file, return default values
  }

  return info;
}

bool
CziMultiFileUtils::IsChildCziFile(const std::string& filepath)
{
  try {
    auto info = GetFileInfo(filepath);
    return info.isChildFile;
  } catch (...) {
    return false;
  }
}

bool
CziMultiFileUtils::IsMultiFileCzi(const std::string& filepath)
{
  try {
    std::filesystem::path fpath(filepath);
    auto stream = libCZI::CreateStreamFromFile(fpath.wstring().c_str());

    auto dirInfo = CziDirectoryParser::ParseDirectory(stream.get());
    return dirInfo.IsMultiFile();

  } catch (...) {
    return false;
  }
}

bool
CziMultiFileUtils::ValidateFileGuid(const std::string& filepath, const libCZI::GUID& expectedPrimaryGuid)
{
  try {
    auto info = GetFileInfo(filepath);
    return CziDirectoryParser::GuidsEqual(info.primaryFileGuid, expectedPrimaryGuid);
  } catch (...) {
    return false;
  }
}

std::vector<std::string>
CziMultiFileUtils::GenerateChildFilePaths(const std::string& parentFilepath, int filePart)
{
  std::filesystem::path parentPath(parentFilepath);
  std::string stem = parentPath.stem().string();
  std::string extension = parentPath.extension().string();
  std::string parentDir = parentPath.parent_path().string();

  std::vector<std::string> candidates;

  // Pattern 1: parent.czi -> parent(1).czi, parent(2).czi
  std::string pattern1 = parentDir + "/" + stem + "(" + std::to_string(filePart) + ")" + extension;
  candidates.push_back(pattern1);

  // Pattern 2: parent.czi -> parent_1.czi, parent_2.czi
  std::string pattern2 = parentDir + "/" + stem + "_" + std::to_string(filePart) + extension;
  candidates.push_back(pattern2);

  // Pattern 3: parent.czi -> parent.czi.001, parent.czi.002
  std::ostringstream ss;
  ss << parentFilepath << "." << std::setw(3) << std::setfill('0') << filePart;
  candidates.push_back(ss.str());

  return candidates;
}

std::optional<std::string>
CziMultiFileUtils::ExtractParentNameFromPattern(const std::string& filename)
{
  // Try pattern: parent(1).czi -> parent.czi
  std::regex pattern1(R"((.+)\(\d+\)\.czi$)");
  std::smatch match;
  if (std::regex_match(filename, match, pattern1)) {
    return match[1].str() + ".czi";
  }

  // Try pattern: parent_1.czi -> parent.czi
  std::regex pattern2(R"((.+)_\d+\.czi$)");
  if (std::regex_match(filename, match, pattern2)) {
    return match[1].str() + ".czi";
  }

  // Try pattern: parent.czi.001 -> parent.czi
  std::regex pattern3(R"((.+\.czi)\.\d{3}$)");
  if (std::regex_match(filename, match, pattern3)) {
    return match[1].str();
  }

  return std::nullopt;
}

std::optional<std::string>
CziMultiFileUtils::FindParentFile(const std::string& childFilepath)
{
  std::filesystem::path childPath(childFilepath);
  std::string filename = childPath.filename().string();

  auto parentName = ExtractParentNameFromPattern(filename);
  if (!parentName) {
    return std::nullopt;
  }

  std::filesystem::path parentPath = childPath.parent_path() / *parentName;

  if (std::filesystem::exists(parentPath)) {
    // Validate it's actually the parent by checking GUID
    try {
      auto childInfo = GetFileInfo(childFilepath);
      if (ValidateFileGuid(parentPath.string(), childInfo.primaryFileGuid)) {
        return parentPath.string();
      }
    } catch (...) {
      // If validation fails, still return the path if it exists
      return parentPath.string();
    }
  }

  return std::nullopt;
}

std::vector<std::string>
CziMultiFileUtils::FindChildFiles(const std::string& parentFilepath)
{
  std::vector<std::string> childFiles;

  try {
    // Get parent file's primary GUID for validation
    auto parentInfo = GetFileInfo(parentFilepath);

    // Parse directory to find max file part
    std::filesystem::path fpath(parentFilepath);
    auto stream = libCZI::CreateStreamFromFile(fpath.wstring().c_str());
    auto dirInfo = CziDirectoryParser::ParseDirectory(stream.get());

    int maxFilePart = dirInfo.GetMaxFilePart();

    // Try to find each child file
    for (int filePart = 1; filePart <= maxFilePart; ++filePart) {
      auto candidates = GenerateChildFilePaths(parentFilepath, filePart);

      for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) {
          // Validate GUID matches
          if (ValidateFileGuid(candidate, parentInfo.primaryFileGuid)) {
            childFiles.push_back(candidate);
            break; // Found this file part, move to next
          }
        }
      }
    }

  } catch (...) {
    // Return whatever we found
  }

  return childFiles;
}

} // namespace czi_multi
