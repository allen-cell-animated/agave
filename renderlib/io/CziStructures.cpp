#include "CziStructures.h"
#include <cstring>

namespace czi_multi {

bool
ByteOrderConverter::IsLittleEndian()
{
  std::uint16_t test = 0x0001;
  return *reinterpret_cast<std::uint8_t*>(&test) == 0x01;
}

std::int16_t
ByteOrderConverter::SwapInt16(std::int16_t val)
{
  return static_cast<std::int16_t>(((val & 0x00FF) << 8) | ((val & 0xFF00) >> 8));
}

std::uint16_t
ByteOrderConverter::SwapUInt16(std::uint16_t val)
{
  return ((val & 0x00FF) << 8) | ((val & 0xFF00) >> 8);
}

std::int32_t
ByteOrderConverter::SwapInt32(std::int32_t val)
{
  return static_cast<std::int32_t>(((val & 0x000000FF) << 24) | ((val & 0x0000FF00) << 8) | ((val & 0x00FF0000) >> 8) |
                                   ((val & 0xFF000000) >> 24));
}

std::uint32_t
ByteOrderConverter::SwapUInt32(std::uint32_t val)
{
  return ((val & 0x000000FF) << 24) | ((val & 0x0000FF00) << 8) | ((val & 0x00FF0000) >> 8) |
         ((val & 0xFF000000) >> 24);
}

std::int64_t
ByteOrderConverter::SwapInt64(std::int64_t val)
{
  return static_cast<std::int64_t>(((val & 0x00000000000000FFLL) << 56) | ((val & 0x000000000000FF00LL) << 40) |
                                   ((val & 0x0000000000FF0000LL) << 24) | ((val & 0x00000000FF000000LL) << 8) |
                                   ((val & 0x000000FF00000000LL) >> 8) | ((val & 0x0000FF0000000000LL) >> 24) |
                                   ((val & 0x00FF000000000000LL) >> 40) | ((val & 0xFF00000000000000LL) >> 56));
}

std::uint64_t
ByteOrderConverter::SwapUInt64(std::uint64_t val)
{
  return ((val & 0x00000000000000FFULL) << 56) | ((val & 0x000000000000FF00ULL) << 40) |
         ((val & 0x0000000000FF0000ULL) << 24) | ((val & 0x00000000FF000000ULL) << 8) |
         ((val & 0x000000FF00000000ULL) >> 8) | ((val & 0x0000FF0000000000ULL) >> 24) |
         ((val & 0x00FF000000000000ULL) >> 40) | ((val & 0xFF00000000000000ULL) >> 56);
}

float
ByteOrderConverter::SwapFloat(float val)
{
  union
  {
    float f;
    std::uint32_t u;
  } data;
  data.f = val;
  data.u = SwapUInt32(data.u);
  return data.f;
}

libCZI::GUID
ByteOrderConverter::SwapGUID(const libCZI::GUID& guid)
{
  libCZI::GUID result;
  result.Data1 = SwapUInt32(guid.Data1);
  result.Data2 = SwapUInt16(guid.Data2);
  result.Data3 = SwapUInt16(guid.Data3);
  std::memcpy(result.Data4, guid.Data4, 8);
  return result;
}

void
ByteOrderConverter::ConvertFileHeaderSegment(FileHeaderSegment& header)
{
  if (IsLittleEndian()) {
    return; // CZI is little-endian, no conversion needed
  }

  header.header.AllocatedSize = SwapInt64(header.header.AllocatedSize);
  header.header.UsedSize = SwapInt64(header.header.UsedSize);
  header.data.Major = SwapInt32(header.data.Major);
  header.data.Minor = SwapInt32(header.data.Minor);
  header.data.Reserved1 = SwapUInt32(header.data.Reserved1);
  header.data.Reserved2 = SwapUInt32(header.data.Reserved2);
  header.data.PrimaryFileGuid = SwapGUID(header.data.PrimaryFileGuid);
  header.data.FileGuid = SwapGUID(header.data.FileGuid);
  header.data.FilePart = SwapInt32(header.data.FilePart);
  header.data.SubBlockDirectoryPosition = SwapInt64(header.data.SubBlockDirectoryPosition);
  header.data.MetadataPosition = SwapInt64(header.data.MetadataPosition);
  header.data.UpdatePending = SwapInt32(header.data.UpdatePending);
  header.data.AttachmentDirectoryPosition = SwapInt64(header.data.AttachmentDirectoryPosition);
}

void
ByteOrderConverter::ConvertSubBlockDirectorySegment(SubBlockDirectorySegment& dir)
{
  if (IsLittleEndian()) {
    return; // CZI is little-endian, no conversion needed
  }

  dir.header.AllocatedSize = SwapInt64(dir.header.AllocatedSize);
  dir.header.UsedSize = SwapInt64(dir.header.UsedSize);
  dir.data.EntryCount = SwapInt32(dir.data.EntryCount);
}

void
ByteOrderConverter::ConvertDimensionEntryDV(DimensionEntryDV& entry)
{
  if (IsLittleEndian()) {
    return;
  }

  entry.Start = SwapInt32(entry.Start);
  entry.Size = SwapInt32(entry.Size);
  entry.StartCoordinate = SwapFloat(entry.StartCoordinate);
  entry.StoredSize = SwapInt32(entry.StoredSize);
}

void
ByteOrderConverter::ConvertSubBlockDirectoryEntryDV(SubBlockDirectoryEntryDV& entry)
{
  if (IsLittleEndian()) {
    return;
  }

  entry.PixelType = SwapInt32(entry.PixelType);
  entry.FilePosition = SwapInt64(entry.FilePosition);
  entry.FilePart = SwapInt32(entry.FilePart);
  entry.Compression = SwapInt32(entry.Compression);
  entry.DimensionCount = SwapInt32(entry.DimensionCount);
}

} // namespace czi_multi
