#include "FileReaderZarr.h"

#include "BoundingBox.h"
#include "ImageXYZC.h"
#include "Logging.h"
#include "StringUtil.h"
#include "VolumeDimensions.h"

#include "tensorstore/context.h"
#include "tensorstore/index_space/dim_expression.h"
#include "tensorstore/kvstore/generation.h"
#include "tensorstore/open.h"

#include <algorithm>
#include <chrono>
#include <map>
#include <set>

static bool
isHTTP(const std::string& filepath)
{
  return filepath.find("http") == 0;
}
static bool
isS3(const std::string& filepath)
{
  return filepath.find("s3:") == 0;
}
static bool
isGS(const std::string& filepath)
{
  return filepath.find("gs:") == 0;
}

static bool
isCloud(const std::string& filepath)
{
  return isHTTP(filepath) || isGS(filepath) || isS3(filepath);
}

static nlohmann::json
getKvStoreDriverParams(const std::string& filepath, const std::string& subpath)
{
  if (isCloud(filepath)) {
    if (isS3(filepath)) {
      // parse bucket and path from s3 url string of the form s3://bucket/path
      std::string bucket = filepath.substr(5);
      size_t pos = bucket.find("/");
      std::string path = bucket.substr(pos + 1);
      bucket = bucket.substr(0, pos);
      if (!endsWith(path, "/")) {
        path += "/";
      }
      path = path + subpath;
      if (!endsWith(path, "/")) {
        path += "/";
      }
      return { { "driver", "s3" }, { "bucket", bucket }, { "path", path } };
    } else if (isGS(filepath)) {
      // parse bucket and path from gs url string of the form gs://bucket/path
      std::string bucket = filepath.substr(5);
      size_t pos = bucket.find("/");
      std::string path = bucket.substr(pos + 1);
      bucket = bucket.substr(0, pos);
      if (!endsWith(path, "/")) {
        path += "/";
      }
      path = path + subpath;
      if (!endsWith(path, "/")) {
        path += "/";
      }
      return { { "driver", "gcs" }, { "bucket", bucket }, { "path", path } };
    } else {
      return { { "driver", "http" }, { "base_url", filepath }, { "path", subpath } };
    }
  } else {
    // if file path does not end with slash then add one
    // TODO maybe use std::filesystem::path for cross-platform?
    std::string path = filepath;
    if (path.back() != '/' && path.back() != '\\') {
      path += "/";
    }
    if (!subpath.empty()) {
      path += subpath;
    }
    return {
      { "driver", "file" },
      { "path", path },
    };
  }
}

FileReaderZarr::FileReaderZarr(const std::string& filepath) {}

FileReaderZarr::~FileReaderZarr() {}

::nlohmann::json
FileReaderZarr::jsonRead(const std::string& zarrurl)
{
  if (m_zattrs.is_object()) {
    return m_zattrs;
  }

  // JSON uses a separate driver
  auto attrs_store_open_result = tensorstore::Open<::nlohmann::json, 0>(
                                   { { "driver", "json" }, { "kvstore", getKvStoreDriverParams(zarrurl, ".zattrs") } })
                                   .result();
  if (!attrs_store_open_result.ok()) {
    LOG_ERROR << "Error: " << attrs_store_open_result.status();
    return ::nlohmann::json::object_t();
  }
  auto attrs_store = attrs_store_open_result.value();
  // Sets attrs_array to a rank-0 array of ::nlohmann::json
  auto attrs_array_result = tensorstore::Read(attrs_store).result();

  ::nlohmann::json attrs;
  if (attrs_array_result.ok()) {
    attrs = attrs_array_result.value()();
    // std::cout << "attrs: " << attrs << std::endl;
  } else {
    LOG_ERROR << "Error: " << attrs_array_result.status();
    if (absl::IsNotFound(attrs_array_result.status())) {
      attrs = ::nlohmann::json::object_t();
    }
  }
  m_zattrs = attrs;
  return attrs;
}

std::vector<std::string>
FileReaderZarr::getChannelNames(const std::string& filepath)
{
  std::vector<std::string> channelNames;
  nlohmann::json attrs = jsonRead(filepath);
  auto omero = m_zattrs["omero"];
  if (omero.is_object()) {
    auto channels = omero["channels"];
    if (channels.is_array()) {
      for (auto& channel : channels) {
        channelNames.push_back(channel["label"]);
      }
    }
  }
  return channelNames;
}
uint32_t
FileReaderZarr::loadNumScenes(const std::string& filepath)
{
  nlohmann::json attrs = jsonRead(filepath);
  auto multiscales = attrs["multiscales"];
  if (multiscales.is_array()) {
    return multiscales.size();
  }
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

std::string
getSpatialUnit(nlohmann::json axes)
{
  std::string unit = "units";
  for (auto axis : axes) {
    // use first spatial axis.
    // identify spatial axis by type=space or name=z,y,x
    std::string type = axis["type"];
    if (type == "space") {
      auto unitobj = axis["unit"];
      if (unitobj.is_string()) {
        unit = unitobj;
        return unit;
      }
    }
    std::string name = axis["name"];
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::toupper(c); });
    if (name == "Z" || name == "Y" || name == "X") {
      auto unitobj = axis["unit"];
      if (unitobj.is_string()) {
        unit = unitobj;
        return unit;
      }
    }
  }
  return unit;
}

std::vector<std::string>
getAxes(nlohmann::json axes)
{
  //"axes": [
  //    {
  //        "name": "t",
  //        "type": "time",
  //        "unit": "millisecond"
  //    },
  //    {
  //        "name": "c",
  //        "type": "channel"
  //    },
  //    {
  //        "name": "z",
  //        "type": "space",
  //        "unit": "micrometer"
  //    },
  //    {
  //        "name": "y",
  //        "type": "space",
  //        "unit": "micrometer"
  //    },
  //    {
  //        "name": "x",
  //        "type": "space",
  //        "unit": "micrometer"
  //    }
  //],

  // is array!
  // we will recognize only t,c,z,y,x...
  // according to spec (https://ngff.openmicroscopy.org/latest/#multiscale-md)
  // this must be of length 2-5
  // and contain only 2 or 3 spatial axes
  // type time must come before type channel, and spatial axes (type=space) must follow
  // therefore:
  // // if we assume zyx preference, then:
  // the last 2 axes must be spatial
  // if there are 2 axes, the order must be yx
  // if there are 5 axes, the order must be tczyx
  // 2 spatial axes:
  // YX    = 111YX -> [0,1,2]
  // TYX   = T11YX -> [1,2]
  // CYX   = 1C1YX -> [0,2]
  // TCYX  = TC1YX -> [2]
  // 3 spatial axes:
  // ZYX   = 11ZYX -> [0,1]
  // TZYX  = T1ZYX -> [1]
  // CZYX  = 1CZYX -> [0]
  // TCZYX = TCZYX -> []
  // are allowed

  // count spatial axes

  std::vector<std::string> dims;
  for (auto axis : axes) {
    std::string name = axis["name"];
    // LOG_INFO << name;
    // convert to uppercase
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::toupper(c); });
    dims.push_back(name);
  }
  return dims;
}

std::vector<MultiscaleDims>
FileReaderZarr::loadMultiscaleDims(const std::string& filepath, uint32_t scene)
{
  std::vector<MultiscaleDims> multiscaleDims;

  nlohmann::json attrs = jsonRead(filepath);

  std::vector<std::string> channelNames = getChannelNames(filepath);

  auto multiscales = attrs["multiscales"];
  if (multiscales.is_array()) {
    auto multiscale = multiscales[scene];
    std::vector<std::string> dimorder = { "T", "C", "Z", "Y", "X" };
    auto axes = multiscale["axes"];
    if (!axes.is_null()) {
      dimorder = getAxes(axes);
    }
    auto datasets = multiscale["datasets"];
    if (datasets.is_array()) {
      for (auto& dataset : datasets) {
        auto path = dataset["path"];
        if (path.is_string()) {
          std::string pathstr = path;
          auto result = tensorstore::Open({ { "driver", "zarr" },
                                            { "kvstore",

                                              getKvStoreDriverParams(filepath, pathstr) } })
                          .result();
          if (!result.ok()) {
            LOG_ERROR << "Error: " << result.status();
            LOG_ERROR << "Failed to open store for " << filepath << " :: " << pathstr;
          } else {
            tensorstore::TensorStore<> store = result.value();
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
              zmd.dimensionOrder = dimorder;
              zmd.scale = scalevec;
              zmd.shape = shape;
              // TODO reconcile these strings against my other ways of specifying dtype
              zmd.dtype = dtype.name();
              zmd.path = pathstr;
              zmd.channelNames = channelNames;
              zmd.spatialUnits = VolumeDimensions::sanitizeUnitsString(getSpatialUnit(axes));
              multiscaleDims.push_back(zmd);
            }
          }
        }
      }
    }
  } else {
    LOG_ERROR << "No multiscales array found in " << filepath;
  }
  return multiscaleDims;
}

VolumeDimensions
FileReaderZarr::loadDimensions(const std::string& filepath, uint32_t scene)
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
FileReaderZarr::loadFromFile(const LoadSpec& loadSpec)
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

  uint32_t nch = loadSpec.channels.empty() ? dims.sizeC : loadSpec.channels.size();

  if (!m_store.valid()) {
    auto context = tensorstore::Context::FromJson({ { "cache_pool", { { "total_bytes_limit", 100000000 } } } }).value();

    auto openFuture = tensorstore::Open(
      { { "driver", "zarr" }, { "kvstore", getKvStoreDriverParams(loadSpec.filepath, loadSpec.subpath) } },
      context,
      tensorstore::OpenMode::open,
      tensorstore::RecheckCached{ false },
      tensorstore::RecheckCachedData{ false },
      tensorstore::ReadWriteMode::read);

    auto result = openFuture.result();
    if (!result.ok()) {
      LOG_ERROR << "Error: " << result.status();
      return emptyimage;
    }

    m_store = result.value();
  }
  auto domain = m_store.domain();
  // std::cout << "domain.shape(): " << domain.shape() << std::endl;
  // std::cout << "domain.origin(): " << domain.origin() << std::endl;
  // auto shape_span = store.domain().shape();

  // std::vector<int64_t> shape(shape_span.begin(), shape_span.end());

  size_t planesize_bytes = dims.sizeX * dims.sizeY * (ImageXYZC::IN_MEMORY_BPP / 8);
  size_t channelsize_bytes = planesize_bytes * dims.sizeZ;
  uint8_t* data = new uint8_t[channelsize_bytes * nch];
  memset(data, 0, channelsize_bytes * nch);
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
  for (uint32_t channel = 0; channel < nch; ++channel) {
    uint32_t channelToLoad = channel;
    if (!loadSpec.channels.empty()) {
      channelToLoad = loadSpec.channels[channel];
    }
    // read entire channel into its native size
    destptr = channelRawMem;

    tensorstore::IndexTransform<> transform = tensorstore::IdentityTransform(m_store.domain());
    // T value:
    // transform = (std::move(transform) | tensorstore::Dims(0).IndexSlice(time)).value();
    // C value:
    // transform = (std::move(transform) | tensorstore::Dims(1).IndexSlice(channel)).value();
    // for (unsigned d = 0; d < shape.size(); ++d) {
    //  transform = (std::move(transform) | tensorstore::Dims(d).HalfOpenInterval(0, shape[d] / 2)).value();
    //}
    // auto x = tensorstore::Read<tensorstore::zero_origin>(store | transform).value();
    // auto* p = reinterpret_cast<uint16_t*>(x.data());

    // make sure this works with 2d, 3d, or 4d data
    int tsdim = 0;
    if (levelDims.hasDim("T")) {
      transform =
        (std::move(transform) | tensorstore::Dims(tsdim).HalfOpenInterval(loadSpec.time, loadSpec.time + 1)).value();
      tsdim++;
    }
    if (levelDims.hasDim("C")) {
      transform =
        (std::move(transform) | tensorstore::Dims(tsdim).HalfOpenInterval(channelToLoad, channelToLoad + 1)).value();
      tsdim++;
    }
    if (levelDims.hasDim("Z")) {
      transform = (std::move(transform) | tensorstore::Dims(tsdim).HalfOpenInterval(minz, maxz)).value();
      tsdim++;
    }
    transform = (std::move(transform) | tensorstore::Dims(tsdim).HalfOpenInterval(miny, maxy)).value();
    tsdim++;
    transform = (std::move(transform) | tensorstore::Dims(tsdim).HalfOpenInterval(minx, maxx)).value();
    tsdim++;

    static constexpr tensorstore::DimensionIndex kNumDims = 5;
    const tensorstore::Index shapeToLoad[kNumDims] = { 1, 1, dims.sizeZ, dims.sizeY, dims.sizeX };
    if (levelDims.dtype == "uint8") {
      auto arr = tensorstore::Array(reinterpret_cast<uint8_t*>(destptr), shapeToLoad);
      tensorstore::Read(m_store | transform, tensorstore::UnownedToShared(arr)).value();
    } else if (levelDims.dtype == "int32") {
      auto arr = tensorstore::Array(reinterpret_cast<int32_t*>(destptr), shapeToLoad);
      tensorstore::Read(m_store | transform, tensorstore::UnownedToShared(arr)).value();
    } else if (levelDims.dtype == "uint16") {
      auto arr = tensorstore::Array(reinterpret_cast<uint16_t*>(destptr), shapeToLoad);
      tensorstore::Read(m_store | transform, tensorstore::UnownedToShared(arr)).value();
    } else if (levelDims.dtype == "float32") {
      auto arr = tensorstore::Array(reinterpret_cast<float*>(destptr), shapeToLoad);
      tensorstore::Read(m_store | transform, tensorstore::UnownedToShared(arr)).value();
    } else {
      LOG_ERROR << "Unrecognized format (" << levelDims.dtype
                << "). Please let us know if you need support for this format. Can not load data.";
      return emptyimage;
    }

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
                                nch,
                                ImageXYZC::IN_MEMORY_BPP, // dims.bitsPerPixel,
                                smartPtr.release(),
                                dims.physicalSizeX,
                                dims.physicalSizeY,
                                dims.physicalSizeZ,
                                dims.spatialUnits);

  std::vector<std::string> channelNames = dims.getChannelNames(loadSpec.channels);
  im->setChannelNames(channelNames);

  tEnd = std::chrono::high_resolution_clock::now();
  elapsed = tEnd - tStartImage;
  LOG_DEBUG << "ImageXYZC prepared in " << (elapsed.count() * 1000.0) << "ms";

  elapsed = tEnd - tStart;
  LOG_DEBUG << "Loaded " << loadSpec.filepath << " in " << (elapsed.count() * 1000.0) << "ms";

  std::shared_ptr<ImageXYZC> sharedImage(im);
  return sharedImage;
}
