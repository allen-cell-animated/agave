#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

class ImageXYZC;
struct VolumeDimensions;
struct MultiscaleDims;

// TODO this is sort of zarr specific
// and if we ever want to load multiscale CZI or TIFF
// then we probably need to generalize this differently
struct LoadSpec
{
  std::string filepath;
  // important for zarr multiscale
  std::string subpath;

  uint32_t scene;
  uint32_t time;
  // set all to 0 to load all x,y,z
  uint32_t minx, maxx, miny, maxy, minz, maxz;

  LoadSpec()
    : scene(0)
    , time(0)
    , minx(0)
    , maxx(0)
    , miny(0)
    , maxy(0)
    , minz(0)
    , maxz(0)
  {
  }

  std::string toString() const;
  size_t getMemoryEstimate() const;
  static std::string bytesToStringLabel(size_t mem);

  std::string getFilename() const;
};

class FileReader
{
public:
  FileReader();
  virtual ~FileReader();

  static uint32_t loadNumScenes(const std::string& filepath);

  // return dimensions from scene in file
  static VolumeDimensions loadFileDimensions(const std::string& filepath, uint32_t scene = 0);
  static bool loadMultiscaleDims(const std::string& filepath, uint32_t scene, std::vector<MultiscaleDims>& dims);

  static std::shared_ptr<ImageXYZC> loadFromFile(const LoadSpec& loadSpec, bool addToCache = false);

  static std::shared_ptr<ImageXYZC> loadFromArray_4D(uint8_t* dataArray,
                                                     std::vector<uint32_t> shape,
                                                     const std::string& name,
                                                     std::vector<char> dims = {},
                                                     std::vector<std::string> channelNames = {},
                                                     std::vector<float> physicalSizes = { 1.0f, 1.0f, 1.0f },
                                                     bool addToCache = false);

private:
  static std::map<std::string, std::shared_ptr<ImageXYZC>> sPreloadedImageCache;
};
