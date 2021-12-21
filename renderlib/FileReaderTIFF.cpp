#include "FileReaderTIFF.h"

#include "BoundingBox.h"
#include "ImageXYZC.h"
#include "Logging.h"
#include "VolumeDimensions.h"

#include "pugixml/pugixml.hpp"

#include <tiff.h>
#include <tiffio.h>

#include <chrono>
#include <map>
#include <set>

static const uint32_t IN_MEMORY_BPP = 16;

FileReaderTIFF::FileReaderTIFF() {}

FileReaderTIFF::~FileReaderTIFF() {}

// TODO: move into a String Utils
static std::string
trim(const std::string& str, const std::string& whitespace = " \t")
{
  const auto strBegin = str.find_first_not_of(whitespace);
  if (strBegin == std::string::npos)
    return ""; // no content

  const auto strEnd = str.find_last_not_of(whitespace);
  const auto strRange = strEnd - strBegin;

  return str.substr(strBegin, strRange);
}

// TODO: move into a String Utils
static bool
startsWith(const std::string mainStr, const std::string toMatch)
{
  // std::string::find returns 0 if toMatch is found at starting
  if (mainStr.find(toMatch) == 0)
    return true;
  else
    return false;
}

// TODO: move into a String Utils
static bool
endsWith(std::string const& value, std::string const& ending)
{
  if (ending.size() > value.size())
    return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

// TODO: move into a String Utils
static void
split(const std::string& s, char delim, std::vector<std::string>& elems)
{
  std::stringstream ss;
  ss.str(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
}

class ScopedTiffReader
{
public:
  ScopedTiffReader(const std::string& filepath)
  {
    // Loads tiff file
    m_tiff = TIFFOpen(filepath.c_str(), "r");
    if (!m_tiff) {
      LOG_ERROR << "Failed to open TIFF: '" << filepath << "'";
    }
  }
  ~ScopedTiffReader()
  {
    if (m_tiff) {
      TIFFClose(m_tiff);
    }
  }
  TIFF* reader() { return m_tiff; }

protected:
  TIFF* m_tiff;
};

uint32_t
requireUint32Attr(pugi_agave::xml_node& el, const std::string& attr, uint32_t defaultVal)
{
  return (uint32_t)el.attribute(attr.c_str()).as_uint(defaultVal);
}

float
requireFloatAttr(pugi_agave::xml_node& el, const std::string& attr, float defaultVal)
{
  return el.attribute(attr.c_str()).as_float(defaultVal);
}

uint32_t
FileReaderTIFF::loadNumScenesTiff(const std::string& filepath)
{
  return 1;
}

bool
readTiffDimensions(TIFF* tiff, const std::string filepath, VolumeDimensions& dims, uint32_t scene)
{
  char* imagedescription = nullptr;
  // metadata is in ImageDescription of first IFD in the file.
  if (TIFFGetField(tiff, TIFFTAG_IMAGEDESCRIPTION, &imagedescription) != 1) {
    imagedescription = nullptr;
    LOG_WARNING << "Failed to read imagedescription of TIFF: '" << filepath << "';  interpreting as single channel.";
  }

  // Temporary variables
  uint32_t width, height;
  //  tsize_t scanlength;

  // Read dimensions of image
  if (TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width) != 1) {
    LOG_ERROR << "Failed to read width of TIFF: '" << filepath << "'";
    return false;
  }
  if (TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height) != 1) {
    LOG_ERROR << "Failed to read height of TIFF: '" << filepath << "'";
    return false;
  }

  uint32_t bpp = 0;
  if (TIFFGetField(tiff, TIFFTAG_BITSPERSAMPLE, &bpp) != 1) {
    LOG_ERROR << "Failed to read bpp of TIFF: '" << filepath << "'";
    return false;
  }

  uint16_t sampleFormat = SAMPLEFORMAT_UINT;
  if (TIFFGetField(tiff, TIFFTAG_SAMPLEFORMAT, &sampleFormat) != 1) {
    LOG_WARNING << "Failed to read sampleformat of TIFF: '" << filepath << "'";
  }
  if (sampleFormat != SAMPLEFORMAT_UINT && sampleFormat != SAMPLEFORMAT_IEEEFP) {
    LOG_ERROR << "Unsupported tiff SAMPLEFORMAT " << sampleFormat << " for '" << filepath << "'";
    return false;
  }

  uint32_t sizeT = 1;
  uint32_t sizeX = width;
  uint32_t sizeY = height;
  uint32_t sizeZ = 1;
  uint32_t sizeC = 1;
  float physicalSizeX = 1.0f;
  float physicalSizeY = 1.0f;
  float physicalSizeZ = 1.0f;
  std::vector<std::string> channelNames;
  std::string dimensionOrder = "XYCZT";

  std::string simagedescription = trim(imagedescription);

  // check for plain tiff with ImageJ imagedescription:
  if (startsWith(simagedescription, "ImageJ=")) {
    // "ImageJ=\nhyperstack=true\nimages=7900\nchannels=1\nslices=50\nframes=158"
    // "ImageJ=1.52i\nimages=126\nchannels=2\nslices=63\nhyperstack=true\nmode=composite\nunit=
    //      micron\nfinterval=299.2315368652344\nspacing=0.2245383462882669\nloop=false\nmin=9768.0\nmax=
    //        14591.0\n"
    std::vector<std::string> sl;
    split(simagedescription, '\n', sl);
    // split each string into name/value pairs,
    // then look up as a map.
    std::map<std::string, std::string> imagejmetadata;
    for (int i = 0; i < sl.size(); ++i) {
      std::vector<std::string> namevalue;
      split(sl[i], '=', namevalue);
      if (namevalue.size() == 2) {
        imagejmetadata[namevalue[0]] = namevalue[1];
      } else if (namevalue.size() == 1) {
        imagejmetadata[namevalue[0]] = "";
      } else {
        LOG_ERROR << "Unexpected name/value pair in TIFF ImageJ metadata: " << sl[i];
        return false;
      }
    }

    auto iter = imagejmetadata.find("channels");
    if (iter != imagejmetadata.end()) {
      sizeC = std::stoi((*iter).second);
    } else {
      LOG_WARNING << "Failed to read number of channels of ImageJ TIFF: '" << filepath << "'";
    }

    iter = imagejmetadata.find("slices");
    if (iter != imagejmetadata.end()) {
      sizeZ = std::stoi((*iter).second);
    } else {
      LOG_WARNING << "Failed to read number of slices of ImageJ TIFF: '" << filepath << "'";
    }

    iter = imagejmetadata.find("frames");
    if (iter != imagejmetadata.end()) {
      sizeT = std::stoi((*iter).second);
    } else {
      LOG_WARNING << "Failed to read number of frames of ImageJ TIFF: '" << filepath << "'";
    }

    iter = imagejmetadata.find("spacing");
    if (iter != imagejmetadata.end()) {
      try {
        physicalSizeZ = std::stof((*iter).second);
        if (physicalSizeZ < 0.0f) {
          physicalSizeZ = -physicalSizeZ;
        }
        physicalSizeX = physicalSizeZ;
        physicalSizeY = physicalSizeZ;
      } catch (...) {
        LOG_WARNING << "Failed to read spacing of ImageJ TIFF: '" << filepath << "'";
        physicalSizeZ = 1.0f;
      }
    }

    for (uint32_t i = 0; i < sizeC; ++i) {
      channelNames.push_back(std::to_string(i));
    }
  } else if (startsWith(simagedescription, "{\"shape\":")) {
    // expect a 4d shape array of C,Z,Y,X or 5d T,C,Z,Y,X
    size_t firstBracket = simagedescription.find_first_of('[');
    size_t lastBracket = simagedescription.find_last_of(']');
    std::string shape = simagedescription.substr(firstBracket + 1, lastBracket - firstBracket - 1);
    LOG_INFO << shape;
    std::vector<std::string> shapelist;
    split(shape, ',', shapelist);
    if ((shapelist.size() != 4) && (shapelist.size() != 5)) {
      LOG_ERROR << "Expected shape to be 4D or 5D TIFF: '" << filepath << "'";
      return false;
    }
    dimensionOrder = "XYZCT";
    bool hasT = (shapelist.size() == 5);
    int shapeIndex = 0;
    if (hasT) {
      sizeT = std::stoi(shapelist[shapeIndex++]);
    }
    sizeC = std::stoi(shapelist[shapeIndex++]);
    sizeZ = std::stoi(shapelist[shapeIndex++]);
    sizeY = std::stoi(shapelist[shapeIndex++]);
    sizeX = std::stoi(shapelist[shapeIndex++]);
    for (uint32_t i = 0; i < sizeC; ++i) {
      channelNames.push_back(std::to_string(i));
    }

  } else if (startsWith(simagedescription, "<?xml version") && endsWith(simagedescription, "OME>")) {
    // convert c to xml doc.  if this fails then we don't have an ome tif.
    pugi_agave::xml_document omexml;
    pugi_agave::xml_parse_result parseOk = omexml.load_string(simagedescription.c_str());
    if (!parseOk) {
      LOG_ERROR << "Bad OME xml metadata content";
      return false;
    }

    pugi_agave::xml_node omenode = omexml.child("OME");

    // extract some necessary info from the xml:

    // count how many <Image> tags and that is our number of scenes.
    uint32_t numScenes = 0;
    pugi_agave::xml_node imageEl;
    pugi_agave::xml_node pixelsEl;
    for (pugi_agave::xml_node imagenode : omenode.children("Image")) {
      // get the Image and Pixels element of scene
      if (numScenes == scene) {
        imageEl = imagenode;
        pixelsEl = imagenode.child("Pixels");
      }
      numScenes++;
    }
    if (scene >= numScenes) {
      LOG_ERROR << "Requested invalid scene index " << scene << " in OME TIFF; returning scene 0";
      scene = 0;
    }

    if (!imageEl || !pixelsEl) {
      LOG_ERROR << "No <Pixels> element in ome xml for scene " << scene;
      return false;
    }

    // skipping "complex", "double-complex", and "bit".
    std::map<std::string, uint32_t> mapPixelTypeBPP = { { "uint8", 8 },  { "uint16", 16 }, { "uint32", 32 },
                                                        { "int8", 8 },   { "int16", 16 },  { "int32", 32 },
                                                        { "float", 32 }, { "double", 64 } };

    std::string pixelType = pixelsEl.attribute("Type").as_string("uint16");
    std::transform(
      pixelType.begin(), pixelType.end(), pixelType.begin(), [](unsigned char c) { return std::tolower(c); });
    LOG_INFO << "pixel type: " << pixelType;
    bpp = mapPixelTypeBPP[pixelType];
    if (bpp != 32 && bpp != 16 && bpp != 8) {
      LOG_ERROR << "Image must be 8 or 16-bit integer, or 32-bit float typed";
      return false;
    }
    sizeX = requireUint32Attr(pixelsEl, "SizeX", 0);
    sizeY = requireUint32Attr(pixelsEl, "SizeY", 0);
    sizeZ = requireUint32Attr(pixelsEl, "SizeZ", 0);
    sizeC = requireUint32Attr(pixelsEl, "SizeC", 0);
    sizeT = requireUint32Attr(pixelsEl, "SizeT", 0);
    // one of : "XYZCT", "XYZTC","XYCTZ","XYCZT","XYTCZ","XYTZC"
    dimensionOrder = pixelsEl.attribute("DimensionOrder").as_string(dimensionOrder.c_str());
    physicalSizeX = requireFloatAttr(pixelsEl, "PhysicalSizeX", 1.0f);
    physicalSizeY = requireFloatAttr(pixelsEl, "PhysicalSizeY", 1.0f);
    physicalSizeZ = requireFloatAttr(pixelsEl, "PhysicalSizeZ", 1.0f);
    std::string physicalSizeXunit = pixelsEl.attribute("PhysicalSizeXUnit").as_string("");
    std::string physicalSizeYunit = pixelsEl.attribute("PhysicalSizeYUnit").as_string("");
    std::string physicalSizeZunit = pixelsEl.attribute("PhysicalSizeZUnit").as_string("");

    // find channel names
    int i = 0;
    for (pugi_agave::xml_node node : pixelsEl.children("Channel")) {
      std::string chid = node.attribute("ID").value();
      std::string chname = node.attribute("Name").value();
      if (!chname.empty()) {
        channelNames.push_back(chname);
      } else if (!chid.empty()) {
        channelNames.push_back(chid);
      } else {
        channelNames.push_back(std::to_string(i));
      }
      i++;
    }
  } else {
    // unrecognized string / no metadata.
    // walk the file and count the directories and assume that is Z
    // sizeZ was initialized to 1.
    sizeZ = 0;
    while (TIFFSetDirectory(tiff, sizeZ)) {
      sizeZ++;
    };
    channelNames.push_back("0");
  }

  assert(sizeX == width);
  assert(sizeY == height);

  // allocate the destination buffer!!!!
  assert(sizeT >= 1);
  assert(sizeC >= 1);
  assert(sizeX >= 1);
  assert(sizeY >= 1);
  assert(sizeZ >= 1);

  dims.sizeX = sizeX;
  dims.sizeY = sizeY;
  dims.sizeZ = sizeZ;
  dims.sizeC = sizeC;
  dims.sizeT = sizeT;
  dims.dimensionOrder = dimensionOrder;
  dims.physicalSizeX = physicalSizeX;
  dims.physicalSizeY = physicalSizeY;
  dims.physicalSizeZ = physicalSizeZ;
  dims.bitsPerPixel = bpp;
  dims.channelNames = channelNames;
  dims.sampleFormat = sampleFormat;

  dims.log();

  return dims.validate();
}

// return number of bytes copied to dest
size_t
copyDirect(uint8_t* dest, const uint8_t* src, size_t numBytes, int srcBitsPerPixel)
{
  memcpy(dest, src, numBytes);
  return numBytes;
}

// convert pixels
// this assumes tight packing of pixels in both buf(source) and dataptr(dest)
// assumes dest is of format IN_MEMORY_BPP
// return 1 for successful conversion, 0 on failure (e.g. unacceptable srcBitsPerPixel)
size_t
convertChannelData(uint8_t* dest, const uint8_t* src, const VolumeDimensions& dims)
{
  // how many pixels in this channel:
  size_t numPixels = dims.sizeX * dims.sizeY * dims.sizeZ;
  int srcBitsPerPixel = dims.bitsPerPixel;

  // dest bits per pixel is IN_MEMORY_BPP which is currently 16, or 2 bytes
  if (IN_MEMORY_BPP == srcBitsPerPixel) {
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

// DANGER: assumes dataPtr has enough space allocated!!!!
bool
readTiffPlane(TIFF* tiff, int planeIndex, const VolumeDimensions& dims, uint8_t* dataPtr)
{
  int setdirok = TIFFSetDirectory(tiff, planeIndex);
  if (setdirok == 0) {
    LOG_ERROR << "Bad tiff directory specified: " << (planeIndex);
    return false;
  }

  tmsize_t numBytesRead = 0;
  // TODO future optimize:
  // This function is usually called in a loop. We could factor out the TIFFmalloc and TIFFfree calls.
  // Should profile to see if the repeated malloc/frees are any kind of loading bottleneck.
  if (TIFFIsTiled(tiff)) {
    tsize_t tilesize = TIFFTileSize(tiff);
    uint32_t ntiles = TIFFNumberOfTiles(tiff);
    if (ntiles != 1) {
      LOG_ERROR << "Reader doesn't support more than 1 tile per plane";
      return false;
    }
    // assuming ntiles == 1 for all IFDs
    tdata_t buf = _TIFFmalloc(tilesize);

    uint32_t i = planeIndex;

    numBytesRead = TIFFReadEncodedTile(tiff, 0, buf, tilesize);
    if (numBytesRead < 0) {
      LOG_ERROR << "Error reading tiff tile";
      _TIFFfree(buf);
      return false;
    }
    // copy buf into data.
    size_t numBytesCopied = copyDirect(dataPtr, static_cast<uint8_t*>(buf), numBytesRead, dims.bitsPerPixel);

    _TIFFfree(buf);
    // if something went wrong at this level, bail out
    if (numBytesCopied == 0) {
      return false;
    }

  } else {
    // stripped.

    uint32_t i = planeIndex;

    uint32_t planeindexintiff = i;

    // Number of bytes in a decoded scanline
    tsize_t striplength = TIFFStripSize(tiff);
    tdata_t buf = _TIFFmalloc(striplength);

    uint32_t nstrips = TIFFNumberOfStrips(tiff);
    // LOG_DEBUG << nstrips;     // num y rows
    // LOG_DEBUG << striplength; // x width * rows per strip
    for (tstrip_t strip = 0; strip < nstrips; strip++) {
      numBytesRead = TIFFReadEncodedStrip(tiff, strip, buf, striplength);
      if (numBytesRead < 0) {
        LOG_ERROR << "Error reading tiff strip";
        _TIFFfree(buf);
        return false;
      }

      // copy buf into data.
      size_t numBytesCopied = copyDirect(dataPtr, static_cast<uint8_t*>(buf), numBytesRead, dims.bitsPerPixel);

      // if something went wrong at this level, bail out
      if (numBytesCopied == 0) {
        _TIFFfree(buf);
        return false;
      }
      // advance to next strip
      dataPtr += numBytesCopied;
    }
    _TIFFfree(buf);
  }
  return true;
}

VolumeDimensions
FileReaderTIFF::loadDimensionsTiff(const std::string& filepath, uint32_t scene)
{
  ScopedTiffReader tiffreader(filepath);
  TIFF* tiff = tiffreader.reader();
  // Loads tiff file
  if (!tiff) {
    return VolumeDimensions();
  }

  VolumeDimensions dims;
  int32_t numScenesInFile = 0;
  bool dims_ok = readTiffDimensions(tiff, filepath, dims, scene);
  if (!dims_ok) {
    return VolumeDimensions();
  }
  return dims;
}

std::shared_ptr<ImageXYZC>
FileReaderTIFF::loadOMETiff(const std::string& filepath, VolumeDimensions* outDims, uint32_t time, uint32_t scene)
{
  std::shared_ptr<ImageXYZC> emptyimage;

  auto tStart = std::chrono::high_resolution_clock::now();

  // Loads tiff file
  ScopedTiffReader tiffreader(filepath);
  TIFF* tiff = tiffreader.reader();
  if (!tiff) {
    return emptyimage;
  }

  VolumeDimensions dims;
  bool dims_ok = readTiffDimensions(tiff, filepath, dims, scene);
  if (!dims_ok) {
    return emptyimage;
  }

  if (scene > 0) {
    LOG_WARNING << "Multiscene tiff not supported yet. Using scene 0";
    scene = 0;
  }
  if (time > (int32_t)(dims.sizeT - 1)) {
    LOG_ERROR << "Time " << time << " exceeds time samples in file: " << dims.sizeT;
    return emptyimage;
  }

  LOG_DEBUG << "Reading " << (TIFFIsTiled(tiff) ? "tiled" : "stripped") << " tiff...";

  uint32_t rowsPerStrip = 0;
  if (TIFFGetField(tiff, TIFFTAG_ROWSPERSTRIP, &rowsPerStrip)) {
    LOG_DEBUG << "ROWSPERSTRIP: " << rowsPerStrip;
    uint32_t StripsPerImage = ((dims.sizeY + rowsPerStrip - 1) / rowsPerStrip);
    LOG_DEBUG << "Strips per image: " << StripsPerImage;
  }
  uint32_t samplesPerPixel = 0;
  if (TIFFGetField(tiff, TIFFTAG_SAMPLESPERPIXEL, &samplesPerPixel)) {
    LOG_DEBUG << "SamplesPerPixel: " << samplesPerPixel;
  }
  if (samplesPerPixel != 1) {
    LOG_ERROR << "" << samplesPerPixel << " samples per pixel is not supported in tiff";
    return emptyimage;
  }

  uint32_t planarConfig = 0;
  if (TIFFGetField(tiff, TIFFTAG_PLANARCONFIG, &planarConfig)) {
    LOG_DEBUG << "PlanarConfig: " << (planarConfig == 1 ? "PLANARCONFIG_CONTIG" : "PLANARCONFIG_SEPARATE");
  }

  size_t planesize_bytes = dims.sizeX * dims.sizeY * (IN_MEMORY_BPP / 8);
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
    for (uint32_t slice = 0; slice < dims.sizeZ; ++slice) {
      uint32_t planeIndex = dims.getPlaneIndex(slice, channel, time);
      destptr = channelRawMem + slice * rawPlanesize;
      if (!readTiffPlane(tiff, planeIndex, dims, destptr)) {
        return emptyimage;
      }
    }

    // convert to our internal format (IN_MEMORY_BPP)
    if (!convertChannelData(data + channel * channelsize_bytes, channelRawMem, dims)) {
      return emptyimage;
    }
  }

  auto tEnd = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = tEnd - tStart;
  LOG_DEBUG << "TIFF loaded in " << (elapsed.count() * 1000.0) << "ms";

  auto tStartImage = std::chrono::high_resolution_clock::now();

  // TODO: convert data to uint16_t pixels if not already.
  // we can release the smartPtr because ImageXYZC will now own the raw data memory
  ImageXYZC* im = new ImageXYZC(dims.sizeX,
                                dims.sizeY,
                                dims.sizeZ,
                                dims.sizeC,
                                IN_MEMORY_BPP, // dims.bitsPerPixel,
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
