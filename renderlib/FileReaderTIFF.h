#pragma once

#include <memory>
#include <string>

class ImageXYZC;

class FileReaderTIFF
{
public:
  FileReaderTIFF();
  virtual ~FileReaderTIFF();

  static std::shared_ptr<ImageXYZC> loadOMETiff_4D(const std::string& filepath);
  // static void tiffReadChannelXYZ(uint8_t* byteptr, const VolumeDimensions& dims, int c, int t = 0);
};
