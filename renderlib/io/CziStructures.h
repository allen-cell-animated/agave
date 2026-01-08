#pragma once

#include <cstdint>
#include <cstring>
#include "libCZI/Src/libCZI/libCZI.h"

// CZI binary file format structures
// These mirror the structures in libCZI but are defined here to allow parsing
// without modifying libCZI itself.

namespace czi_multi {

static constexpr size_t SIZE_FILEHEADER_DATA = 512;

#pragma pack(push, 1)

// File header segment at offset 0
struct FileHeaderSegment
{
  struct
  {
    char Id[16]; // "ZISRAWFILE\0\0\0\0\0\0"
    std::int64_t AllocatedSize;
    std::int64_t UsedSize;
  } header;

  struct
  {
    std::int32_t Major;
    std::int32_t Minor;
    std::uint32_t Reserved1;
    std::uint32_t Reserved2;
    libCZI::GUID PrimaryFileGuid; // GUID linking multi-file set
    libCZI::GUID FileGuid;        // GUID of this file
    std::int32_t FilePart;        // 0=parent, 1+=child
    std::int64_t SubBlockDirectoryPosition;
    std::int64_t MetadataPosition;
    std::int32_t UpdatePending;
    std::int64_t AttachmentDirectoryPosition;
    unsigned char Spare[SIZE_FILEHEADER_DATA - 80];
  } data;

  static constexpr size_t SIZE_FILEHEADER_DATA = 512;
};

// Subblock directory segment
struct SubBlockDirectorySegment
{
  struct
  {
    char Id[16]; // "ZISRAWDIRECTORY\0"
    std::int64_t AllocatedSize;
    std::int64_t UsedSize;
  } header;

  struct
  {
    std::int32_t EntryCount;
    unsigned char Reserved[124];
  } data;
};

// Dimension entry for subblock directory entries
struct DimensionEntryDV
{
  char Dimension[4];
  std::int32_t Start;
  std::int32_t Size;
  float StartCoordinate;
  std::int32_t StoredSize;
};

// Subblock directory entry with variable dimensions (DV schema)
struct SubBlockDirectoryEntryDV
{
  char SchemaType[2]; // "DV"
  std::int32_t PixelType;
  std::int64_t FilePosition; // Offset in file specified by FilePart
  std::int32_t FilePart;     // Which file contains this subblock
  std::int32_t Compression;
  std::uint8_t PyramidType; // spare[0]
  std::uint8_t Spare1;
  std::uint8_t Spare2;
  std::uint8_t Spare3;
  std::uint8_t Spare4;
  std::uint8_t Spare5;
  std::int32_t DimensionCount;
  // Followed by DimensionCount x DimensionEntryDV
};

// Subblock directory entry with dimension entries (DE schema)
struct SubBlockDirectoryEntryDE
{
  char SchemaType[2]; // "DE"
  std::int32_t PixelType;
  std::int64_t FilePosition; // Offset in file specified by FilePart
  std::int32_t FilePart;     // Which file contains this subblock
  std::int32_t Compression;
  std::uint8_t PyramidType;
  std::uint8_t Spare1;
  std::uint8_t Spare2;
  std::uint8_t Spare3;
  std::uint8_t Spare4;
  std::uint8_t Spare5;
  char DimensionEntries[32]; // Fixed dimension entries
                             // Variable size based on content
};

#pragma pack(pop)

// Byte order conversion utilities
class ByteOrderConverter
{
public:
  static void ConvertFileHeaderSegment(FileHeaderSegment& header);
  static void ConvertSubBlockDirectorySegment(SubBlockDirectorySegment& dir);
  static void ConvertDimensionEntryDV(DimensionEntryDV& entry);
  static void ConvertSubBlockDirectoryEntryDV(SubBlockDirectoryEntryDV& entry);

private:
  static std::int16_t SwapInt16(std::int16_t val);
  static std::uint16_t SwapUInt16(std::uint16_t val);
  static std::int32_t SwapInt32(std::int32_t val);
  static std::uint32_t SwapUInt32(std::uint32_t val);
  static std::int64_t SwapInt64(std::int64_t val);
  static std::uint64_t SwapUInt64(std::uint64_t val);
  static float SwapFloat(float val);
  static libCZI::GUID SwapGUID(const libCZI::GUID& guid);
  static bool IsLittleEndian();
};

} // namespace czi_multi
