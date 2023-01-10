#include "FileReaderZarr.h"

#include "BoundingBox.h"
#include "ImageXYZC.h"
#include "Logging.h"
#include "VolumeDimensions.h"

#include "tensorstore/context.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/open.h"

#include <algorithm>
#include <chrono>
#include <map>
#include <set>

FileReaderZarr::FileReaderZarr() {}

FileReaderZarr::~FileReaderZarr() {}

static ::nlohmann::json
jsonRead(std::string zarrurl)
{
  // JSON uses a separate driver
  auto attrs_store =
    tensorstore::Open<::nlohmann::json, 0>(
      { { "driver", "json" }, { "kvstore", { { "driver", "http" }, { "base_url", zarrurl }, { "path", ".zattrs" } } } })
      .result()
      .value();

  // Sets attrs_array to a rank-0 array of ::nlohmann::json
  auto attrs_array_result = tensorstore::Read(attrs_store).result();

  ::nlohmann::json attrs;
  if (attrs_array_result.ok()) {
    attrs = attrs_array_result.value()();
    std::cout << "attrs: " << attrs << std::endl;
  } else if (absl::IsNotFound(attrs_array_result.status())) {
    attrs = ::nlohmann::json::object_t();
  } else {
    std::cout << "Error: " << attrs_array_result.status();
  }
  return attrs;
}

uint32_t
FileReaderZarr::loadNumScenesZarr(const std::string& filepath)
{
  return 1;
}

// return number of bytes copied to dest
static size_t
copyDirect(uint8_t* dest, const uint8_t* src, size_t numBytes, int srcBitsPerPixel)
{
  memcpy(dest, src, numBytes);
  return numBytes;
}

// convert pixels
// this assumes tight packing of pixels in both buf(source) and dataptr(dest)
// assumes dest is of format IN_MEMORY_BPP
// return 1 for successful conversion, 0 on failure (e.g. unacceptable srcBitsPerPixel)
static size_t
convertChannelData(uint8_t* dest, const uint8_t* src, const VolumeDimensions& dims)
{
  // how many pixels in this channel:
  size_t numPixels = dims.sizeX * dims.sizeY * dims.sizeZ;
  int srcBitsPerPixel = dims.bitsPerPixel;

  // dest bits per pixel is IN_MEMORY_BPP which is currently 16, or 2 bytes
  if (ImageXYZC::IN_MEMORY_BPP == srcBitsPerPixel) {
    memcpy(dest, src, numPixels * (srcBitsPerPixel / 8));
    return 1;
  } else if (srcBitsPerPixel == 8) {
    uint16_t* dataptr16 = reinterpret_cast<uint16_t*>(dest);
    for (size_t b = 0; b < numPixels; ++b) {
      *dataptr16 = (uint16_t)src[b];
      dataptr16++;
    }
    return 1;
  } else if (srcBitsPerPixel == 32) {
    // assumes 32-bit floating point (not int or uint)
    uint16_t* dataptr16 = reinterpret_cast<uint16_t*>(dest);
    const float* src32 = reinterpret_cast<const float*>(src);
    // compute min and max; and then rescale values to fill dynamic range.
    float lowest = FLT_MAX;
    float highest = -FLT_MAX;
    float f;
    for (size_t b = 0; b < numPixels; ++b) {
      f = src32[b];
      if (f < lowest) {
        lowest = f;
      }
      if (f > highest) {
        highest = f;
      }
    }
    for (size_t b = 0; b < numPixels; ++b) {
      *dataptr16 = (uint16_t)((src32[b] - lowest) / (highest - lowest) * 65535.0);
      dataptr16++;
    }
    return 1;
  } else {
    LOG_ERROR << "Unexpected tiff pixel size " << srcBitsPerPixel << " bits";
    return 0;
  }
  return 0;
}

std::vector<MultiscaleDims>
FileReaderZarr::loadMultiscaleDims(const std::string& filepath, uint32_t scene)
{
  std::vector<MultiscaleDims> multiscaleDims;

  nlohmann::json attrs = jsonRead(filepath);
  auto multiscales = attrs["multiscales"];
  if (multiscales.is_array()) {
    // take the first one for now.
    auto multiscale = multiscales[0];
    auto datasets = multiscale["datasets"];
    if (datasets.is_array()) {
      for (auto& dataset : datasets) {
        auto path = dataset["path"];
        if (path.is_string()) {
          std::string pathstr = path;
          auto store = tensorstore::Open(
                         { { "driver", "zarr" },
                           { "kvstore", { { "driver", "http" }, { "base_url", filepath }, { "path", pathstr } } } })
                         .result()
                         .value();
          tensorstore::DataType dtype = store.dtype();
          auto shape_span = store.domain().shape();
          std::cout << "Level " << multiscaleDims.size() << " shape " << shape_span << std::endl;
          std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

          auto scale = dataset["coordinateTransformations"][0]["scale"];
          if (scale.is_array()) {
            std::vector<float> scalevec;
            for (auto& s : scale) {
              scalevec.push_back(s);
            }
            MultiscaleDims zmd;
            zmd.scale = scalevec;
            zmd.shape = shape;
            // TODO reconcile these strings against my other ways of specifying dtype
            zmd.dtype = dtype.name();
            zmd.path = pathstr;
            multiscaleDims.push_back(zmd);
          }
        }
      }
    }
  }
  return multiscaleDims;
}

VolumeDimensions
FileReaderZarr::loadDimensionsZarr(const std::string& filepath, uint32_t scene)
{
  VolumeDimensions dims;

  // pre-fetch dims for the different multiscales
  std::vector<MultiscaleDims> multiscaleDims;
  multiscaleDims = loadMultiscaleDims(filepath, scene);

  // select a mltiscale level here!
  int level = multiscaleDims.size() - 1;
  MultiscaleDims levelDims = multiscaleDims[level];
  dims = levelDims.getVolumeDimensions();

  dims.log();

  if (!dims.validate()) {
    return VolumeDimensions();
  }

  return dims;
}

std::shared_ptr<ImageXYZC>
FileReaderZarr::loadOMEZarr(const LoadSpec& loadSpec)
{
  auto tStart = std::chrono::high_resolution_clock::now();
  // load channels
  std::shared_ptr<ImageXYZC> emptyimage;

  // pre-fetch dims for the different multiscales
  std::vector<MultiscaleDims> multiscaleDims;
  multiscaleDims = loadMultiscaleDims(loadSpec.filepath, loadSpec.scene);
  if (multiscaleDims.size() < 1) {
    return emptyimage;
  }
  // find loadspec subpath in multiscaledims:
  auto it = std::find_if(multiscaleDims.begin(), multiscaleDims.end(), [&](const MultiscaleDims& md) {
    return md.path == loadSpec.subpath;
  });
  if (it == multiscaleDims.end()) {
    LOG_ERROR << "Could not find subpath " << loadSpec.subpath << " in multiscaleDims";
    return emptyimage;
  }
  MultiscaleDims levelDims = *it;

  VolumeDimensions dims = levelDims.getVolumeDimensions();
  if (loadSpec.maxx > loadSpec.minx)
    dims.sizeX = loadSpec.maxx - loadSpec.minx;
  if (loadSpec.maxy > loadSpec.miny)
    dims.sizeY = loadSpec.maxy - loadSpec.miny;
  if (loadSpec.maxz > loadSpec.minz)
    dims.sizeZ = loadSpec.maxz - loadSpec.minz;

  tensorstore::Context context = tensorstore::Context::Default();

  auto openFuture = tensorstore::Open(
    {
      { "driver", "zarr" },
      { "kvstore",
        { { "driver", "http" },
          { "base_url", loadSpec.filepath },
          { "path", loadSpec.subpath /* dims.zarrSubpath */ } } },
    },
    context,
    tensorstore::OpenMode::open,
    tensorstore::RecheckCached{ false },
    tensorstore::ReadWriteMode::read);

  auto result = openFuture.result();
  if (!result.ok()) {
    return emptyimage;
  }

  auto store = result.value();
  auto domain = store.domain();
  std::cout << "domain.shape(): " << domain.shape() << std::endl;
  std::cout << "domain.origin(): " << domain.origin() << std::endl;
  auto shape_span = store.domain().shape();

  std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

  size_t planesize_bytes = dims.sizeX * dims.sizeY * (ImageXYZC::IN_MEMORY_BPP / 8);
  size_t channelsize_bytes = planesize_bytes * dims.sizeZ;
  uint8_t* data = new uint8_t[channelsize_bytes * dims.sizeC];
  memset(data, 0, channelsize_bytes * dims.sizeC);
  // stash it here in case of early exit, it will be deleted
  std::unique_ptr<uint8_t[]> smartPtr(data);

  uint8_t* destptr = data;

  // still assuming 1 sample per pixel (scalar data) here.
  size_t rawPlanesize = dims.sizeX * dims.sizeY * (dims.bitsPerPixel / 8);
  // allocate temp data for one channel
  uint8_t* channelRawMem = new uint8_t[dims.sizeZ * rawPlanesize];
  memset(channelRawMem, 0, dims.sizeZ * rawPlanesize);

  // stash it here in case of early exit, it will be deleted
  std::unique_ptr<uint8_t[]> smartPtrTemp(channelRawMem);
  uint32_t minx, maxx, miny, maxy, minz, maxz;
  minx = (loadSpec.maxx > loadSpec.minx) ? loadSpec.minx : 0;
  miny = (loadSpec.maxy > loadSpec.miny) ? loadSpec.miny : 0;
  minz = (loadSpec.maxz > loadSpec.minz) ? loadSpec.minz : 0;
  maxx = (loadSpec.maxx > loadSpec.minx) ? loadSpec.maxx : dims.sizeX;
  maxy = (loadSpec.maxy > loadSpec.miny) ? loadSpec.maxy : dims.sizeY;
  maxz = (loadSpec.maxz > loadSpec.minz) ? loadSpec.maxz : dims.sizeZ;
  if (dims.sizeZ != maxz - minz) {
    LOG_ERROR << "Zarr: sizeZ mismatch: " << dims.sizeZ << " vs " << maxz - minz;
  }
  if (dims.sizeY != maxy - miny) {
    LOG_ERROR << "Zarr: sizeY mismatch: " << dims.sizeY << " vs " << maxy - miny;
  }
  if (dims.sizeX != maxx - minx) {
    LOG_ERROR << "Zarr: sizeX mismatch: " << dims.sizeX << " vs " << maxx - minx;
  }

  // now ready to read channels one by one.
  for (uint32_t channel = 0; channel < dims.sizeC; ++channel) {
    // read entire channel into its native size
    //    for (uint32_t slice = 0; slice < dims.sizeZ; ++slice) {
    //    uint32_t planeIndex = dims.getPlaneIndex(0, channel, time);
    destptr = channelRawMem;

    tensorstore::IndexTransform<> transform = tensorstore::IdentityTransform(store.domain());
    // T value:
    // transform = (std::move(transform) | tensorstore::Dims(0).IndexSlice(time)).value();
    // C value:
    // transform = (std::move(transform) | tensorstore::Dims(1).IndexSlice(channel)).value();
    // for (unsigned d = 0; d < shape.size(); ++d) {
    //  transform = (std::move(transform) | tensorstore::Dims(d).HalfOpenInterval(0, shape[d] / 2)).value();
    //}
    // auto x = tensorstore::Read<tensorstore::zero_origin>(store | transform).value();
    // auto* p = reinterpret_cast<uint16_t*>(x.data());

    transform =
      (std::move(transform) | tensorstore::Dims(0).HalfOpenInterval(loadSpec.time, loadSpec.time + 1)).value();
    transform = (std::move(transform) | tensorstore::Dims(1).HalfOpenInterval(channel, channel + 1)).value();
    transform = (std::move(transform) | tensorstore::Dims(2).HalfOpenInterval(minz, maxz)).value();
    transform = (std::move(transform) | tensorstore::Dims(3).HalfOpenInterval(miny, maxy)).value();
    transform = (std::move(transform) | tensorstore::Dims(4).HalfOpenInterval(minx, maxx)).value();

    auto arr = tensorstore::Array(
      reinterpret_cast<uint16_t*>(destptr), { 1, 1, dims.sizeZ, dims.sizeY, dims.sizeX }, tensorstore::c_order);
    tensorstore::Read(store | transform, tensorstore::UnownedToShared(arr)).value();

    //      if (!readTiffPlane(tiff, planeIndex, dims, destptr)) {
    //      return emptyimage;
    //  }
    //}

    // convert to our internal format (IN_MEMORY_BPP)
    if (!convertChannelData(data + channel * channelsize_bytes, channelRawMem, dims)) {
      return emptyimage;
    }
  }

  auto tEnd = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = tEnd - tStart;
  LOG_DEBUG << "zarr loaded in " << (elapsed.count() * 1000.0) << "ms";

  auto tStartImage = std::chrono::high_resolution_clock::now();

  // TODO: convert data to uint16_t pixels if not already.
  // we can release the smartPtr because ImageXYZC will now own the raw data memory
  ImageXYZC* im = new ImageXYZC(dims.sizeX,
                                dims.sizeY,
                                dims.sizeZ,
                                dims.sizeC,
                                ImageXYZC::IN_MEMORY_BPP, // dims.bitsPerPixel,
                                smartPtr.release(),
                                dims.physicalSizeX,
                                dims.physicalSizeY,
                                dims.physicalSizeZ);

  im->setChannelNames(dims.channelNames);

  tEnd = std::chrono::high_resolution_clock::now();
  elapsed = tEnd - tStartImage;
  LOG_DEBUG << "ImageXYZC prepared in " << (elapsed.count() * 1000.0) << "ms";

  elapsed = tEnd - tStart;
  LOG_DEBUG << "Loaded " << loadSpec.filepath << " in " << (elapsed.count() * 1000.0) << "ms";

  std::shared_ptr<ImageXYZC> sharedImage(im);
  return sharedImage;
}

std::shared_ptr<ImageXYZC>
FileReaderZarr::loadOMEZarr(const std::string& filepath, VolumeDimensions* outDims, uint32_t time, uint32_t scene)
{
  auto tStart = std::chrono::high_resolution_clock::now();
  // load channels
  std::shared_ptr<ImageXYZC> emptyimage;

  VolumeDimensions dims = FileReaderZarr::loadDimensionsZarr(filepath, scene);
  if (!dims.validate()) {
    return emptyimage;
  }

  tensorstore::Context context = tensorstore::Context::Default();

  auto openFuture = tensorstore::Open(
    {
      { "driver", "zarr" },
      { "kvstore", { { "driver", "http" }, { "base_url", filepath }, { "path", dims.zarrSubpath } } },
    },
    context,
    tensorstore::OpenMode::open,
    tensorstore::RecheckCached{ false },
    tensorstore::ReadWriteMode::read);

  auto result = openFuture.result();
  if (!result.ok()) {
    return emptyimage;
  }

  auto store = result.value();
  auto domain = store.domain();
  std::cout << "domain.shape(): " << domain.shape() << std::endl;
  std::cout << "domain.origin(): " << domain.origin() << std::endl;
  auto shape_span = store.domain().shape();

  std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

  size_t planesize_bytes = dims.sizeX * dims.sizeY * (ImageXYZC::IN_MEMORY_BPP / 8);
  size_t channelsize_bytes = planesize_bytes * dims.sizeZ;
  uint8_t* data = new uint8_t[channelsize_bytes * dims.sizeC];
  memset(data, 0, channelsize_bytes * dims.sizeC);
  // stash it here in case of early exit, it will be deleted
  std::unique_ptr<uint8_t[]> smartPtr(data);

  uint8_t* destptr = data;

  // still assuming 1 sample per pixel (scalar data) here.
  size_t rawPlanesize = dims.sizeX * dims.sizeY * (dims.bitsPerPixel / 8);
  // allocate temp data for one channel
  uint8_t* channelRawMem = new uint8_t[dims.sizeZ * rawPlanesize];
  memset(channelRawMem, 0, dims.sizeZ * rawPlanesize);

  // stash it here in case of early exit, it will be deleted
  std::unique_ptr<uint8_t[]> smartPtrTemp(channelRawMem);

  // now ready to read channels one by one.
  for (uint32_t channel = 0; channel < dims.sizeC; ++channel) {
    // read entire channel into its native size
    //    for (uint32_t slice = 0; slice < dims.sizeZ; ++slice) {
    //    uint32_t planeIndex = dims.getPlaneIndex(0, channel, time);
    destptr = channelRawMem;

    tensorstore::IndexTransform<> transform = tensorstore::IdentityTransform(store.domain());
    // T value:
    // transform = (std::move(transform) | tensorstore::Dims(0).IndexSlice(time)).value();
    // C value:
    // transform = (std::move(transform) | tensorstore::Dims(1).IndexSlice(channel)).value();
    // for (unsigned d = 0; d < shape.size(); ++d) {
    //  transform = (std::move(transform) | tensorstore::Dims(d).HalfOpenInterval(0, shape[d] / 2)).value();
    //}
    // auto x = tensorstore::Read<tensorstore::zero_origin>(store | transform).value();
    // auto* p = reinterpret_cast<uint16_t*>(x.data());

    transform = (std::move(transform) | tensorstore::Dims(0).HalfOpenInterval(time, time + 1)).value();
    transform = (std::move(transform) | tensorstore::Dims(1).HalfOpenInterval(channel, channel + 1)).value();
    transform = (std::move(transform) | tensorstore::Dims(2).HalfOpenInterval(0, shape[2])).value();
    transform = (std::move(transform) | tensorstore::Dims(3).HalfOpenInterval(0, shape[3])).value();
    transform = (std::move(transform) | tensorstore::Dims(4).HalfOpenInterval(0, shape[4])).value();

    auto arr = tensorstore::Array(
      reinterpret_cast<uint16_t*>(destptr), { 1, 1, dims.sizeZ, dims.sizeY, dims.sizeX }, tensorstore::c_order);
    tensorstore::Read(store | transform, tensorstore::UnownedToShared(arr)).value();

    //      if (!readTiffPlane(tiff, planeIndex, dims, destptr)) {
    //      return emptyimage;
    //  }
    //}

    // convert to our internal format (IN_MEMORY_BPP)
    if (!convertChannelData(data + channel * channelsize_bytes, channelRawMem, dims)) {
      return emptyimage;
    }
  }

  auto tEnd = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = tEnd - tStart;
  LOG_DEBUG << "zarr loaded in " << (elapsed.count() * 1000.0) << "ms";

  auto tStartImage = std::chrono::high_resolution_clock::now();

  // TODO: convert data to uint16_t pixels if not already.
  // we can release the smartPtr because ImageXYZC will now own the raw data memory
  ImageXYZC* im = new ImageXYZC(dims.sizeX,
                                dims.sizeY,
                                dims.sizeZ,
                                dims.sizeC,
                                ImageXYZC::IN_MEMORY_BPP, // dims.bitsPerPixel,
                                smartPtr.release(),
                                dims.physicalSizeX,
                                dims.physicalSizeY,
                                dims.physicalSizeZ);

  im->setChannelNames(dims.channelNames);

  tEnd = std::chrono::high_resolution_clock::now();
  elapsed = tEnd - tStartImage;
  LOG_DEBUG << "ImageXYZC prepared in " << (elapsed.count() * 1000.0) << "ms";

  elapsed = tEnd - tStart;
  LOG_DEBUG << "Loaded " << filepath << " in " << (elapsed.count() * 1000.0) << "ms";

  std::shared_ptr<ImageXYZC> sharedImage(im);
  if (outDims != nullptr) {
    *outDims = dims;
  }
  return sharedImage;
}