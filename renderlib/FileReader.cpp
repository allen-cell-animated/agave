#include "FileReader.h"

#include "FileReaderCCP4.h"
#include "FileReaderCzi.h"
#include "FileReaderTIFF.h"
#include "ImageXYZC.h"
#include "Logging.h"

//#define ENABLE_S3_SDK
//#include "netcdf.h"

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

  if (extstr == ".tif" || extstr == ".tiff") {
    return FileReaderTIFF::loadNumScenesTiff(filepath);
  } else if (extstr == ".czi") {
    return FileReaderCzi::loadNumScenesCzi(filepath);
  } else if (extstr == ".map" || extstr == ".mrc") {
    return FileReaderCCP4::loadNumScenesCCP4(filepath);
  }
  return 0;
}

VolumeDimensions
FileReader::loadFileDimensions(const std::string& filepath, uint32_t scene)
{
  std::string extstr = getExtension(filepath);

  // int ncid;
  // int ok =
  //   nc_open("https://animatedcell-test-data.s3.us-west-2.amazonaws.com/AICS-12_881.zarr/Image_0", NC_NOWRITE, &ncid);

  if (extstr == ".tif" || extstr == ".tiff") {
    return FileReaderTIFF::loadDimensionsTiff(filepath, scene);
  } else if (extstr == ".czi") {
    return FileReaderCzi::loadDimensionsCzi(filepath, scene);
  } else if (extstr == ".map" || extstr == ".mrc") {
    return FileReaderCCP4::loadDimensionsCCP4(filepath, scene);
  } else if (extstr == ".zarr") {
    // int ncid;
    // int ok =
    //   nc_open("https://animatedcell-test-data.s3.us-west-2.amazonaws.com/AICS-12_881.zarr/Image_0", NC_NOWRITE,
    //   &ncid);
    // nc_create("file://./CBCT.zarr#mode=nczarr,file", NC_CLOBBER, &ncid);
    //  return FileReaderZarr::loadDimensionsZarr(filepath, scene);
  }
  return VolumeDimensions();
}

std::shared_ptr<ImageXYZC>
FileReader::loadFromFile(const std::string& filepath,
                         VolumeDimensions* dims,
                         uint32_t time,
                         uint32_t scene,
                         bool addToCache)
{
  // check cache first of all.
  auto cached = sPreloadedImageCache.find(filepath);
  if (cached != sPreloadedImageCache.end()) {
    return cached->second;
  }

  std::shared_ptr<ImageXYZC> image;

  std::string extstr = getExtension(filepath);

  // int ncid;
  // int ok = nc_open("https://animatedcell-test-data.s3.us-west-2.amazonaws.com/AICS-12_881.zarr/Image_0#mode=zarr,s3",
  //                  NC_NOWRITE,
  //                  &ncid);

  if (extstr == ".tif" || extstr == ".tiff") {
    image = FileReaderTIFF::loadOMETiff(filepath, dims, time, scene);
  } else if (extstr == ".czi") {
    image = FileReaderCzi::loadCzi(filepath, dims, time, scene);
  } else if (extstr == ".map" || extstr == ".mrc") {
    image = FileReaderCCP4::loadCCP4(filepath, dims, time, scene);
  }

  if (addToCache && image) {
    sPreloadedImageCache[filepath] = image;
  }
  return image;
}

std::shared_ptr<ImageXYZC>
FileReader::loadFromFile_4D(const std::string& filepath, VolumeDimensions* dims, bool addToCache)
{
  return FileReader::loadFromFile(filepath, dims, 0, 0, addToCache);
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
