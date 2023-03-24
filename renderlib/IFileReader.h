#pragma once

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
  // (TODO should store multiscale index instead?  ...and then find subpath from metadata)
  std::string subpath;

  uint32_t scene;
  uint32_t time;
  // if empty, load all channels
  std::vector<uint32_t> channels;
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

  // GPU estimate for 4 channels and one time at 16bpp
  size_t getMemoryEstimate() const;

  static std::string bytesToStringLabel(size_t mem, int decimals = 1);

  std::string getFilename() const;
};

class IFileReader
{
public:
  virtual ~IFileReader() = default;

  // return true if this reader can load sub-chunks in XYZ
  virtual bool supportChunkedLoading() const = 0;

  // find number of scenes
  virtual uint32_t loadNumScenes(const std::string& filepath) = 0;

  // return dimensions from scene in file
  virtual VolumeDimensions loadDimensions(const std::string& filepath, uint32_t scene = 0) = 0;

  // return dimensions from scene in file
  virtual std::vector<MultiscaleDims> loadMultiscaleDims(const std::string& filepath, uint32_t scene = 0) = 0;

  // load image data from file
  virtual std::shared_ptr<ImageXYZC> loadFromFile(const LoadSpec& loadSpec) = 0;
};
