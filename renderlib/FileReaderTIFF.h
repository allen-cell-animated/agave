#pragma once

#include <memory>
#include <string>

class ImageXYZC;

class FileReaderTIFF
{
public:
  FileReaderTIFF();
  virtual ~FileReaderTIFF();

  static std::shared_ptr<ImageXYZC> loadOMETiff(const std::string& filepath, int32_t time = 0, int32_t scene = 0);
  // static void tiffReadChannelXYZ(uint8_t* byteptr, const VolumeDimensions& dims, int c, int t = 0);
};
