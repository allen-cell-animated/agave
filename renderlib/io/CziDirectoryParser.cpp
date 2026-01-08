#include "CziDirectoryParser.h"
#include <cstring>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <set>

namespace czi_multi {

static const char FILEHDR_MAGIC[16] = { 'Z', 'I', 'S',  'R',  'A',  'W',  'F',  'I',
                                        'L', 'E', '\0', '\0', '\0', '\0', '\0', '\0' };

static const char DIRECTORY_MAGIC[16] = { 'Z', 'I', 'S', 'R', 'A', 'W', 'D', 'I',
                                          'R', 'E', 'C', 'T', 'O', 'R', 'Y', '\0' };

FileHeaderSegment
CziDirectoryParser::ReadFileHeader(libCZI::IStream* stream)
{
  FileHeaderSegment header;
  std::uint64_t bytesRead = 0;

  stream->Read(0, &header, sizeof(header), &bytesRead);

  if (bytesRead != sizeof(header)) {
    throw std::runtime_error("Failed to read file header: incomplete read");
  }

  // Verify magic number
  if (std::memcmp(header.header.Id, FILEHDR_MAGIC, 16) != 0) {
    throw std::runtime_error("Invalid CZI file: bad magic number");
  }

  // Convert byte order if needed
  ByteOrderConverter::ConvertFileHeaderSegment(header);

  return header;
}

SubBlockDirectorySegment
CziDirectoryParser::ReadDirectorySegment(libCZI::IStream* stream, std::uint64_t position)
{

  SubBlockDirectorySegment dirSegment;
  std::uint64_t bytesRead = 0;

  stream->Read(position, &dirSegment, sizeof(dirSegment), &bytesRead);

  if (bytesRead != sizeof(dirSegment)) {
    throw std::runtime_error("Failed to read directory segment: incomplete read");
  }

  // Verify magic number
  if (std::memcmp(dirSegment.header.Id, DIRECTORY_MAGIC, 16) != 0) {
    throw std::runtime_error("Invalid directory segment: bad magic number");
  }

  // Convert byte order if needed
  ByteOrderConverter::ConvertSubBlockDirectorySegment(dirSegment);

  return dirSegment;
}

std::uint64_t
CziDirectoryParser::ParseDVEntry(libCZI::IStream* stream, std::uint64_t position, CziDirectoryInfo& info)
{

  SubBlockDirectoryEntryDV entry;
  std::uint64_t bytesRead = 0;

  // Read the fixed part of the entry
  stream->Read(position, &entry, sizeof(entry), &bytesRead);

  if (bytesRead != sizeof(entry)) {
    throw std::runtime_error("Failed to read DV directory entry");
  }

  // Convert byte order
  ByteOrderConverter::ConvertSubBlockDirectoryEntryDV(entry);

  // Store the file position -> file part mapping
  info.filePositionToFilePart[entry.FilePosition] = entry.FilePart;
  info.uniqueFileParts.insert(entry.FilePart);

  // Calculate size of this entry (fixed part + dimensions)
  std::uint64_t entrySize = sizeof(SubBlockDirectoryEntryDV) + entry.DimensionCount * sizeof(DimensionEntryDV);

  return position + entrySize;
}

std::uint64_t
CziDirectoryParser::ParseDEEntry(libCZI::IStream* stream, std::uint64_t position, CziDirectoryInfo& info)
{

  // Read schema type and basic fields
  struct DEHeader
  {
    char SchemaType[2];
    std::int32_t PixelType;
    std::int64_t FilePosition;
    std::int32_t FilePart;
    std::int32_t Compression;
    std::uint8_t Spare[6];
  } deHeader;

  std::uint64_t bytesRead = 0;
  stream->Read(position, &deHeader, sizeof(deHeader), &bytesRead);

  if (bytesRead != sizeof(deHeader)) {
    throw std::runtime_error("Failed to read DE directory entry");
  }

  // Convert byte order (manually for this structure)
  if (!ByteOrderConverter::IsLittleEndian()) {
    deHeader.PixelType = ByteOrderConverter::SwapInt32(deHeader.PixelType);
    deHeader.FilePosition = ByteOrderConverter::SwapInt64(deHeader.FilePosition);
    deHeader.FilePart = ByteOrderConverter::SwapInt32(deHeader.FilePart);
    deHeader.Compression = ByteOrderConverter::SwapInt32(deHeader.Compression);
  }

  // Store the file position -> file part mapping
  info.filePositionToFilePart[deHeader.FilePosition] = deHeader.FilePart;
  info.uniqueFileParts.insert(deHeader.FilePart);

  // DE entries have fixed size (schema type + fixed dimension encoding)
  // The exact size depends on dimension encoding, but we can estimate
  // For simplicity, read the dimension entries field size
  std::uint64_t entrySize = sizeof(deHeader) + 32; // DimensionEntries field

  return position + entrySize;
}

void
CziDirectoryParser::ParseDirectoryEntries(libCZI::IStream* stream,
                                          std::uint64_t basePosition,
                                          std::int32_t entryCount,
                                          CziDirectoryInfo& info)
{

  std::uint64_t currentPosition = basePosition;

  for (std::int32_t i = 0; i < entryCount; ++i) {
    // Read schema type to determine entry type
    char schemaType[2];
    std::uint64_t bytesRead = 0;
    stream->Read(currentPosition, schemaType, 2, &bytesRead);

    if (bytesRead != 2) {
      throw std::runtime_error("Failed to read schema type");
    }

    if (schemaType[0] == 'D' && schemaType[1] == 'V') {
      currentPosition = ParseDVEntry(stream, currentPosition, info);
    } else if (schemaType[0] == 'D' && schemaType[1] == 'E') {
      currentPosition = ParseDEEntry(stream, currentPosition, info);
    } else {
      // Unknown schema type - skip
      throw std::runtime_error("Unknown directory entry schema type");
    }
  }
}

CziDirectoryInfo
CziDirectoryParser::ParseDirectory(libCZI::IStream* stream)
{
  CziDirectoryInfo info;

  // Read file header
  FileHeaderSegment header = ReadFileHeader(stream);

  info.primaryFileGuid = header.data.PrimaryFileGuid;
  info.fileGuid = header.data.FileGuid;
  info.filePart = header.data.FilePart;

  // Check if directory position is valid
  if (header.data.SubBlockDirectoryPosition == 0 || header.data.SubBlockDirectoryPosition == -1) {
    // No directory present
    info.entryCount = 0;
    return info;
  }

  // Read directory segment
  SubBlockDirectorySegment dirSegment = ReadDirectorySegment(stream, header.data.SubBlockDirectoryPosition);

  info.entryCount = dirSegment.data.EntryCount;

  if (info.entryCount == 0) {
    return info;
  }

  // Parse directory entries
  std::uint64_t entriesStart = header.data.SubBlockDirectoryPosition + sizeof(SubBlockDirectorySegment);
  ParseDirectoryEntries(stream, entriesStart, info.entryCount, info);

  return info;
}

bool
CziDirectoryParser::GuidsEqual(const libCZI::GUID& a, const libCZI::GUID& b)
{
  return a.Data1 == b.Data1 && a.Data2 == b.Data2 && a.Data3 == b.Data3 && std::memcmp(a.Data4, b.Data4, 8) == 0;
}

std::string
CziDirectoryParser::GuidToString(const libCZI::GUID& guid)
{
  std::ostringstream ss;
  ss << std::hex << std::uppercase << std::setfill('0');
  ss << "{";
  ss << std::setw(8) << guid.Data1 << "-";
  ss << std::setw(4) << guid.Data2 << "-";
  ss << std::setw(4) << guid.Data3 << "-";
  ss << std::setw(2) << static_cast<int>(guid.Data4[0]);
  ss << std::setw(2) << static_cast<int>(guid.Data4[1]) << "-";
  for (int i = 2; i < 8; ++i) {
    ss << std::setw(2) << static_cast<int>(guid.Data4[i]);
  }
  ss << "}";
  return ss.str();
}

} // namespace czi_multi
