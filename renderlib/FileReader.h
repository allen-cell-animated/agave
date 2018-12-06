#pragma once

#include <map>
#include <memory>
#include <string>

#include "assimp/Importer.hpp"

class CBoundingBox;
class ImageXYZC;

class FileReader
{
public:
  FileReader();
  virtual ~FileReader();

  static std::shared_ptr<ImageXYZC> loadOMETiff_4D(const std::string& filepath, bool addToCache = false);

  static Assimp::Importer* loadAsset(const char* path, CBoundingBox* bb);

private:
  static std::map<std::string, std::shared_ptr<ImageXYZC>> sPreloadedImageCache;
};
