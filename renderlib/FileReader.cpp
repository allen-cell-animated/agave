#include "FileReader.h"

#include "FileReaderCCP4.h"
#include "FileReaderCzi.h"
#include "FileReaderTIFF.h"
#include "FileReaderZarr.h"
#include "ImageXYZC.h"
#include "Logging.h"

#include <chrono>
#include <filesystem>
#include <map>

std::map<std::string, std::shared_ptr<ImageXYZC>> FileReader::sPreloadedImageCache;

// return file extension as lowercase
std::string
getExtension(const std::string filepath)
{
  std::filesystem::path fpath(filepath);

  std::filesystem::path ext = fpath.extension();
  std::string extstr = ext.string();
  for (std::string::size_type i = 0; i < extstr.length(); ++i) {
    extstr[i] = std::tolower(extstr[i]);
  }

  return extstr;
}

FileReader::FileReader() {}

FileReader::~FileReader() {}

uint32_t
FileReader::loadNumScenes(const std::string& filepath)
{
  std::string extstr = getExtension(filepath);

  if (filepath.find("http") == 0) {
    return FileReaderZarr::loadNumScenesZarr(filepath);
  } else if (extstr == ".tif" || extstr == ".tiff") {
    return FileReaderTIFF::loadNumScenesTiff(filepath);
  } else if (extstr == ".czi") {
    return FileReaderCzi::loadNumScenesCzi(filepath);
  } else if (extstr == ".map" || extstr == ".mrc") {
    return FileReaderCCP4::loadNumScenesCCP4(filepath);
  } else if (extstr == ".zarr") {
    return FileReaderZarr::loadNumScenesZarr(filepath);
  }
  return 0;
}

VolumeDimensions
FileReader::loadFileDimensions(const std::string& filepath, uint32_t scene)
{
  std::string extstr = getExtension(filepath);

  if (filepath.find("http") == 0) {
    return FileReaderZarr::loadDimensionsZarr(filepath, scene);
  } else if (extstr == ".tif" || extstr == ".tiff") {
    return FileReaderTIFF::loadDimensionsTiff(filepath, scene);
  } else if (extstr == ".czi") {
    return FileReaderCzi::loadDimensionsCzi(filepath, scene);
  } else if (extstr == ".map" || extstr == ".mrc") {
    return FileReaderCCP4::loadDimensionsCCP4(filepath, scene);
  } else if (extstr == ".zarr") {
    return FileReaderZarr::loadDimensionsZarr(filepath, scene);
  }
  return VolumeDimensions();
}

bool
FileReader::loadMultiscaleDims(const std::string& filepath, uint32_t scene, std::vector<MultiscaleDims>& dims)
{
  std::vector<MultiscaleDims> loadedDims;

  std::string extstr = getExtension(filepath);

  if (filepath.find("http") == 0) {
    loadedDims = FileReaderZarr::loadMultiscaleDims(filepath, scene);
  } else if (extstr == ".tif" || extstr == ".tiff") {
    loadedDims = FileReaderTIFF::loadMultiscaleDims(filepath, scene);
  } else if (extstr == ".czi") {
    loadedDims = FileReaderCzi::loadMultiscaleDims(filepath, scene);
  } else if (extstr == ".map" || extstr == ".mrc") {
    loadedDims = FileReaderCCP4::loadMultiscaleDims(filepath, scene);
  } else if (extstr == ".zarr") {
    loadedDims = FileReaderZarr::loadMultiscaleDims(filepath, scene);
  }

  if (loadedDims.size() > 0) {
    dims = loadedDims;
    return true;
  }
  return false;
}

std::shared_ptr<ImageXYZC>
FileReader::loadFromFile(const LoadSpec& loadSpec, bool addToCache)
{
  // check cache first of all.
  auto cached = sPreloadedImageCache.find(loadSpec.filepath);
  if (cached != sPreloadedImageCache.end()) {
    return cached->second;
  }

  std::string filepath = loadSpec.filepath;
  VolumeDimensions dims;
  uint32_t time = loadSpec.time;
  uint32_t scene = loadSpec.scene;

  std::shared_ptr<ImageXYZC> image;

  std::string extstr = getExtension(filepath);

  if (filepath.find("http") == 0) {
    image = FileReaderZarr::loadOMEZarr(loadSpec);
  } else if (extstr == ".tif" || extstr == ".tiff") {
    image = FileReaderTIFF::loadOMETiff(loadSpec.filepath, &dims, loadSpec.time, loadSpec.scene);
  } else if (extstr == ".czi") {
    image = FileReaderCzi::loadCzi(loadSpec.filepath, &dims, loadSpec.time, loadSpec.scene);
  } else if (extstr == ".map" || extstr == ".mrc") {
    image = FileReaderCCP4::loadCCP4(loadSpec.filepath, &dims, loadSpec.time, loadSpec.scene);
  } else if (extstr == ".zarr") {
    image = FileReaderZarr::loadOMEZarr(loadSpec.filepath, &dims, loadSpec.time, loadSpec.scene);
  }

  if (addToCache && image) {
    sPreloadedImageCache[filepath] = image;
  }

  return image;
}

std::shared_ptr<ImageXYZC>
FileReader::loadFromArray_4D(uint8_t* dataArray,
                             std::vector<uint32_t> shape,
                             const std::string& name,
                             std::vector<char> dims,
                             std::vector<std::string> channelNames,
                             std::vector<float> physicalSizes,
                             bool addToCache)
{
  // check cache first of all.
  auto cached = sPreloadedImageCache.find(name);
  if (cached != sPreloadedImageCache.end()) {
    return cached->second;
  }

  // assume data is in CZYX order:
  static const int XDIM = 3, YDIM = 2, ZDIM = 1, CDIM = 0;

  size_t ndim = shape.size();
  assert(ndim == 4);

  uint32_t bpp = 16;
  uint32_t sizeT = 1;
  uint32_t sizeX = shape[XDIM];
  uint32_t sizeY = shape[YDIM];
  uint32_t sizeZ = shape[ZDIM];
  uint32_t sizeC = shape[CDIM];
  assert(physicalSizes.size() == 3);
  float physicalSizeX = physicalSizes[0];
  float physicalSizeY = physicalSizes[1];
  float physicalSizeZ = physicalSizes[2];

  // product of all shape elements must equal number of elements in dataArray
  // dims must either be empty or must be of same length as shape, and end in (Y, X), and start with CZ or ZC or Z ?

  auto startTime = std::chrono::high_resolution_clock::now();

  // note that im will take ownership of dataArray
  ImageXYZC* im =
    new ImageXYZC(sizeX, sizeY, sizeZ, sizeC, uint32_t(bpp), dataArray, physicalSizeX, physicalSizeY, physicalSizeZ);

  auto endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = endTime - startTime;
  LOG_DEBUG << "ImageXYZC prepared in " << (elapsed.count() * 1000.0) << "ms";

  im->setChannelNames(channelNames);

  std::shared_ptr<ImageXYZC> sharedImage(im);
  if (addToCache) {
    sPreloadedImageCache[name] = sharedImage;
  }
  return sharedImage;
}

size_t
LoadSpec::getMemoryEstimate() const
{
  size_t npix = 1;
  npix *= (maxx - minx);
  npix *= (maxy - miny);
  npix *= (maxz - minz);
  // on gpu we upload only 4 channels max
  size_t bytesperpixel = 4 * ImageXYZC::IN_MEMORY_BPP / 8; // 4 channels * 2 bytes per channel
  size_t mem = npix * bytesperpixel;                       // overflow?
  return mem;
}

std::string
LoadSpec::bytesToStringLabel(size_t mem)
{
  static const std::vector<std::string> levels = { "B", "KB", "MB", "GB", "TB", "PB" };
  double memvalue = mem;
  int level = 0;
  while (memvalue > 1024.0 && level < levels.size() - 1) {
    memvalue = memvalue / 1024.0;
    level++;
  }

  std::stringstream stream;
  stream << std::fixed << std::setprecision(4) << memvalue;
  stream << " " << levels[level];
  std::string s = stream.str();

  return s;
}

std::string
LoadSpec::toString() const
{
  std::stringstream stream;
  stream << filepath;
  if (!subpath.empty()) {
    stream << " " << subpath;
  }
  stream << " : scene " << scene << " time " << time;
  stream << " X:[" << minx << "," << maxx << "] Y[" << miny << "," << maxy << "] Z[" << minz << "," << maxz << "]";
  std::string s = stream.str();
  return s;
}
