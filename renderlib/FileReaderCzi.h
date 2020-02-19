#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

class CBoundingBox;
class ImageXYZC;

class FileReaderCzi
{
public:
  FileReaderCzi();
  virtual ~FileReaderCzi();

  static std::shared_ptr<ImageXYZC> loadCzi_4D(const std::string& filepath);
};
