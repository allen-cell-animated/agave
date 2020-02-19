#include "FileReader.h"

#include "FileReaderCzi.h"
#include "FileReaderTIFF.h"
#include "ImageXYZC.h"
#include "Logging.h"

#include <QElapsedTimer>

#include <boost/filesystem.hpp>

#include <map>

std::map<std::string, std::shared_ptr<ImageXYZC>> FileReader::sPreloadedImageCache;

FileReader::FileReader() {}

FileReader::~FileReader() {}

std::shared_ptr<ImageXYZC>
FileReader::loadFromFile_4D(const std::string& filepath, bool addToCache)
{
  // check cache first of all.
  auto cached = sPreloadedImageCache.find(filepath);
  if (cached != sPreloadedImageCache.end()) {
    return cached->second;
  }

  std::shared_ptr<ImageXYZC> image;

  boost::filesystem::path fpath(filepath);

  boost::filesystem::path ext = fpath.extension();
  std::string extstr = ext.string();
  for (std::string::size_type i = 0; i < extstr.length(); ++i) {
    extstr[i] = std::tolower(extstr[i]);
  }

  if (extstr == ".tif" || extstr == ".tiff") {
    image = FileReaderTIFF::loadOMETiff_4D(filepath);
  } else if (extstr == ".czi") {
    image = FileReaderCzi::loadCzi_4D(filepath);
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

  QElapsedTimer timer;
  timer.start();

  // note that im will take ownership of dataArray
  ImageXYZC* im =
    new ImageXYZC(sizeX, sizeY, sizeZ, sizeC, uint32_t(bpp), dataArray, physicalSizeX, physicalSizeY, physicalSizeZ);
  LOG_DEBUG << "ImageXYZC prepared in " << timer.elapsed() << "ms";

  im->setChannelNames(channelNames);

  std::shared_ptr<ImageXYZC> sharedImage(im);
  if (addToCache) {
    sPreloadedImageCache[name] = sharedImage;
  }
  return sharedImage;
}
