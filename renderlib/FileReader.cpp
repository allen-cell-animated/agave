#include "FileReader.h"

#include "BoundingBox.h"
#include "ImageXYZC.h"
#include "Logging.h"

#include <QDomDocument>
#include <QElapsedTimer>
#include <QString>
#include <QtDebug>

#include <tiff.h>
#include <tiffio.h>

#include <map>
#include <set>

std::map<std::string, std::shared_ptr<ImageXYZC>> FileReader::sPreloadedImageCache;

bool
startsWith(std::string mainStr, std::string toMatch)
{
  // std::string::find returns 0 if toMatch is found at starting
  if (mainStr.find(toMatch) == 0)
    return true;
  else
    return false;
}

bool
VolumeDimensions::validate() const
{
  bool ok = true;
  if (dimensionOrder == "") {
    LOG_ERROR << "Dimension order is null";
    ok = false;
  }
  if (!startsWith(dimensionOrder, "XY") && !startsWith(dimensionOrder, "YX")) {
    LOG_ERROR << "Invalid dimension order: " << dimensionOrder;
    ok = false;
  }
  // legal characters in DimensionOrder:
  static const std::set<char> standardDims = { 'X', 'Y', 'Z', 'C', 'T' };
  // check for any dups or extra
  std::set<char> dimsInDimOrder;
  bool dups = false;
  bool badDims = false;
  for (auto d : dimensionOrder) {
    if (dimsInDimOrder.count(d) == 0) {
      dimsInDimOrder.insert(d);
    } else {
      dups = true;
    }
    if (standardDims.count(d) == 0) {
      badDims = true;
    }
  }
  if (dups || badDims) {
    LOG_ERROR << "Invalid dimension order: " << dimensionOrder;
    ok = false;
  }

  if (sizeX <= 0) {
    LOG_ERROR << "Invalid X size: " << sizeX;
    ok = false;
  }
  if (sizeY <= 0) {
    LOG_ERROR << "Invalid Y size: " << sizeY;
    ok = false;
  }
  if (sizeZ <= 0) {
    LOG_ERROR << "Invalid Z size: " << sizeZ;
    ok = false;
  }
  if (sizeC <= 0) {
    LOG_ERROR << "Invalid C size: " << sizeC;
    ok = false;
  }
  if (sizeT <= 0) {
    LOG_ERROR << "Invalid T size: " << sizeT;
    ok = false;
  }

  if (physicalSizeX <= 0) {
    LOG_ERROR << "Invalid physical pixel size x: " << physicalSizeX;
    ok = false;
  }
  if (physicalSizeY <= 0) {
    LOG_ERROR << "Invalid physical pixel size y: " << physicalSizeY;
    ok = false;
  }
  if (physicalSizeZ <= 0) {
    LOG_ERROR << "Invalid physical pixel size z: " << physicalSizeZ;
    ok = false;
  }

  if (!channelNames.empty() && channelNames.size() != sizeC) {
    LOG_ERROR << "Invalid number of channel names: " << channelNames.size() << " for " << sizeC << " channels";
    ok = false;
  }

  return ok;
}

uint32_t
VolumeDimensions::getPlaneIndex(uint32_t z, uint32_t c, uint32_t t) const
{
  size_t iz = dimensionOrder.find('Z') - 2;
  size_t ic = dimensionOrder.find('C') - 2;
  size_t it = dimensionOrder.find('T') - 2;
  // assuming dims.validate() == true :
  // assert (iz < 0 || iz > 2 || ic < 0 || ic > 2 || it < 0 || it > 2);

  // check SizeZ
  if (z < 0 || z >= sizeZ) {
    LOG_ERROR << "Invalid Z index: " << z << "/" << sizeZ;
  }

  // check SizeC
  if (c < 0 || c >= sizeC) {
    LOG_ERROR << "Invalid C index: " << c << "/" << sizeC;
  }

  // check SizeT
  if (t < 0 || t >= sizeT) {
    LOG_ERROR << "Invalid T index: " << t << "/" << sizeT;
  }

  // assign rasterization order
  int v0 = iz == 0 ? z : (ic == 0 ? c : t);
  int v1 = iz == 1 ? z : (ic == 1 ? c : t);
  int v2 = iz == 2 ? z : (ic == 2 ? c : t);
  int len0 = iz == 0 ? sizeZ : (ic == 0 ? sizeC : sizeT);
  int len1 = iz == 1 ? sizeZ : (ic == 1 ? sizeC : sizeT);

  return v0 + v1 * len0 + v2 * len0 * len1;
}

std::vector<uint32_t>
VolumeDimensions::getPlaneZCT(uint32_t planeIndex) const
{
  size_t iz = dimensionOrder.find('Z') - 2;
  size_t ic = dimensionOrder.find('C') - 2;
  size_t it = dimensionOrder.find('T') - 2;
  // assuming dims.validate() == true :
  // assert (iz < 0 || iz > 2 || ic < 0 || ic > 2 || it < 0 || it > 2);

  // check image count
  if (planeIndex < 0 || planeIndex >= sizeZ * sizeC * sizeT) {
    LOG_ERROR << "Invalid image index: " << planeIndex << "/" << (sizeZ * sizeC * sizeT);
  }

  // assign rasterization order
  int len0 = iz == 0 ? sizeZ : (ic == 0 ? sizeC : sizeT);
  int len1 = iz == 1 ? sizeZ : (ic == 1 ? sizeC : sizeT);
  // int len2 = iz == 2 ? sizeZ : (ic == 2 ? sizeC : sizeT);
  int v0 = planeIndex % len0;
  int v1 = planeIndex / len0 % len1;
  int v2 = planeIndex / len0 / len1;
  uint32_t z = iz == 0 ? v0 : (iz == 1 ? v1 : v2);
  uint32_t c = ic == 0 ? v0 : (ic == 1 ? v1 : v2);
  uint32_t t = it == 0 ? v0 : (it == 1 ? v1 : v2);

  return { z, c, t };
}

FileReader::FileReader() {}

FileReader::~FileReader() {}

uint32_t
requireUint32Attr(QDomElement& el, const QString& attr, uint32_t defaultVal)
{
  QString attrval = el.attribute(attr);
  bool ok;
  uint32_t retval = attrval.toUInt(&ok);
  if (!ok) {
    retval = defaultVal;
  }
  return retval;
}
float
requireFloatAttr(QDomElement& el, const QString& attr, float defaultVal)
{
  QString attrval = el.attribute(attr);
  bool ok;
  float retval = attrval.toFloat(&ok);
  if (!ok) {
    retval = defaultVal;
  }
  return retval;
}

bool
readTiffDimensions(TIFF* tiff, const std::string filepath, VolumeDimensions& dims)
{
  char* omexmlstr = nullptr;
  // ome-xml is in ImageDescription of first IFD in the file.
  if (TIFFGetField(tiff, TIFFTAG_IMAGEDESCRIPTION, &omexmlstr) != 1) {
    QString msg = "Failed to read imagedescription of TIFF: '" + QString(filepath.c_str()) + "'";
    LOG_ERROR << msg.toStdString();
    return false;
  }

  // Temporary variables
  uint32 width, height;
  //  tsize_t scanlength;

  // Read dimensions of image
  if (TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width) != 1) {
    QString msg = "Failed to read width of TIFF: '" + QString(filepath.c_str()) + "'";
    LOG_ERROR << msg.toStdString();
    return false;
  }
  if (TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height) != 1) {
    QString msg = "Failed to read height of TIFF: '" + QString(filepath.c_str()) + "'";
    LOG_ERROR << msg.toStdString();
    return false;
  }

  uint32_t bpp = 0;
  if (TIFFGetField(tiff, TIFFTAG_BITSPERSAMPLE, &bpp) != 1) {
    QString msg = "Failed to read bpp of TIFF: '" + QString(filepath.c_str()) + "'";
    LOG_ERROR << msg.toStdString();
    return false;
  }

  uint16_t sampleFormat = SAMPLEFORMAT_UINT;
  if (TIFFGetField(tiff, TIFFTAG_SAMPLEFORMAT, &sampleFormat) != 1) {
    // just warn here.  We are not yet using sampleFormat!
    QString msg = "Failed to read sampleformat of TIFF: '" + QString(filepath.c_str()) + "'";
    LOG_WARNING << msg.toStdString();
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
  QString dimensionOrder = "XYCZT";

  // convert to QString for convenience functions
  QString qomexmlstr(omexmlstr);

  // check for plain tiff with ImageJ imagedescription:
  if (qomexmlstr.startsWith("ImageJ=")) {
    // "ImageJ=\nhyperstack=true\nimages=7900\nchannels=1\nslices=50\nframes=158"
    // "ImageJ=1.52i\nimages=126\nchannels=2\nslices=63\nhyperstack=true\nmode=composite\nunit=
    //      micron\nfinterval=299.2315368652344\nspacing=0.2245383462882669\nloop=false\nmin=9768.0\nmax=
    //        14591.0\n"
    QStringList sl = qomexmlstr.split('\n');
    // split each string into name/value pairs,
    // then look up as a map.
    QMap<QString, QString> imagejmetadata;
    for (int i = 0; i < sl.size(); ++i) {
      QStringList namevalue = sl.at(i).split('=');
      if (namevalue.size() == 2) {
        imagejmetadata.insert(namevalue[0], namevalue[1]);
      } else if (namevalue.size() == 1) {
        imagejmetadata.insert(namevalue[0], "");
      } else {
        QString msg = "Unexpected name/value pair in TIFF ImageJ metadata: " + sl.at(i);
        LOG_ERROR << msg.toStdString();
        return false;
      }
    }

    if (imagejmetadata.contains("channels")) {
      QString value = imagejmetadata.value("channels");
      sizeC = value.toInt();
    } else {
      QString msg = "Failed to read number of channels of ImageJ TIFF: '" + QString(filepath.c_str()) + "'";
      LOG_WARNING << msg.toStdString();
    }

    if (imagejmetadata.contains("slices")) {
      QString value = imagejmetadata.value("slices");
      sizeZ = value.toInt();
    } else {
      QString msg = "Failed to read number of slices of ImageJ TIFF: '" + QString(filepath.c_str()) + "'";
      LOG_WARNING << msg.toStdString();
    }

    if (imagejmetadata.contains("spacing")) {
      QString value = imagejmetadata.value("spacing");
      bool ok;
      physicalSizeZ = value.toFloat(&ok);
      if (!ok) {
        QString msg = "Failed to read spacing of ImageJ TIFF: '" + QString(filepath.c_str()) + "'";
        LOG_WARNING << msg.toStdString();
        physicalSizeZ = 1.0f;
      } else {
        if (physicalSizeZ < 0.0f) {
          physicalSizeZ = -physicalSizeZ;
        }
        // WHY?  NEED TO FIGURE OUT UNITS HERE.
        physicalSizeX = 0.1f;
        physicalSizeY = 0.1f;
      }
    }

    for (uint32_t i = 0; i < sizeC; ++i) {
      channelNames.push_back(QString::number(i).toStdString());
    }
  } else if (qomexmlstr.startsWith("{\"shape\":")) {
    // expect a 4d shape array of C,Z,Y,X
    int firstBracket = qomexmlstr.indexOf('[');
    int lastBracket = qomexmlstr.lastIndexOf(']');
    QString shape = qomexmlstr.mid(firstBracket + 1, lastBracket - firstBracket - 1);
    LOG_INFO << shape.toStdString();
    QStringList shapelist = shape.split(',');
    assert(shapelist.size() == 4);
    if (shapelist.size() != 4) {
      QString msg = "Expected shape to be 4D TIFF: '" + QString(filepath.c_str()) + "'";
      LOG_ERROR << msg.toStdString();
      return false;
    }
    dimensionOrder = "XYZCT";
    sizeX = shapelist[3].toInt();
    sizeY = shapelist[2].toInt();
    sizeZ = shapelist[1].toInt();
    sizeC = shapelist[0].toInt();
    for (uint32_t i = 0; i < sizeC; ++i) {
      channelNames.push_back(QString("%1").arg(i).toStdString());
    }

  } else if (qomexmlstr.startsWith("<?xml version") && qomexmlstr.endsWith("OME>")) {
    // convert c to xml doc.  if this fails then we don't have an ome tif.
    QDomDocument omexml;
    bool ok = omexml.setContent(qomexmlstr);
    if (!ok) {
      QString msg = "Bad ome xml content";
      LOG_ERROR << msg.toStdString();
      return false;
    }

    // extract some necessary info from the xml:
    QDomElement pixelsEl = omexml.elementsByTagName("Pixels").at(0).toElement();
    if (pixelsEl.isNull()) {
      QString msg = "No <Pixels> element in ome xml";
      LOG_ERROR << msg.toStdString();
      return false;
    }

    // skipping "complex", "double-complex", and "bit".
    std::map<std::string, uint32_t> mapPixelTypeBPP = { { "uint8", 8 },  { "uint16", 16 }, { "uint32", 32 },
                                                        { "int8", 8 },   { "int16", 16 },  { "int32", 32 },
                                                        { "float", 32 }, { "double", 64 } };

    QString pixelType = pixelsEl.attribute("Type", "uint16").toLower();
    LOG_INFO << "pixel type: " << pixelType.toStdString();
    bpp = mapPixelTypeBPP[pixelType.toStdString()];
    if (bpp != 16) {
      LOG_ERROR << "Image must be 16-bit";
      return false;
    }
    sizeX = requireUint32Attr(pixelsEl, "SizeX", 0);
    sizeY = requireUint32Attr(pixelsEl, "SizeY", 0);
    sizeZ = requireUint32Attr(pixelsEl, "SizeZ", 0);
    sizeC = requireUint32Attr(pixelsEl, "SizeC", 0);
    sizeT = requireUint32Attr(pixelsEl, "SizeT", 0);
    // one of : "XYZCT", "XYZTC","XYCTZ","XYCZT","XYTCZ","XYTZC"
    dimensionOrder = pixelsEl.attribute("DimensionOrder", dimensionOrder);
    physicalSizeX = requireFloatAttr(pixelsEl, "PhysicalSizeX", 1.0f);
    physicalSizeY = requireFloatAttr(pixelsEl, "PhysicalSizeY", 1.0f);
    physicalSizeZ = requireFloatAttr(pixelsEl, "PhysicalSizeZ", 1.0f);
    QString physicalSizeXunit = pixelsEl.attribute("PhysicalSizeXUnit", "");
    QString physicalSizeYunit = pixelsEl.attribute("PhysicalSizeYUnit", "");
    QString physicalSizeZunit = pixelsEl.attribute("PhysicalSizeZUnit", "");

    // find channel names
    QDomNodeList channels = omexml.elementsByTagName("Channel");
    for (int i = 0; i < channels.length(); ++i) {
      QDomNode dn = channels.at(i);
      QDomElement chel = dn.toElement();
      QString chid = chel.attribute("ID");
      QString chname = chel.attribute("Name");
      if (!chname.isEmpty()) {
        channelNames.push_back(chname.toStdString());
      } else if (!chid.isEmpty()) {
        channelNames.push_back(chid.toStdString());
      } else {
        channelNames.push_back(QString("%1").arg(i).toStdString());
      }
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
  assert(sizeC >= 1);
  assert(sizeX >= 1);
  assert(sizeY >= 1);
  assert(sizeZ >= 1);

  dims.sizeX = sizeX;
  dims.sizeY = sizeY;
  dims.sizeZ = sizeZ;
  dims.sizeC = sizeC;
  dims.sizeT = sizeT;
  dims.dimensionOrder = dimensionOrder.toStdString();
  dims.physicalSizeX = physicalSizeX;
  dims.physicalSizeY = physicalSizeY;
  dims.physicalSizeZ = physicalSizeZ;
  dims.bitsPerPixel = bpp;
  dims.channelNames = channelNames;

  return dims.validate();
}

// DANGER: assumes dataPtr has enough space allocated!!!!
bool
readTiffPlane(TIFF* tiff, int planeIndex, uint8_t* dataPtr)
{
  // TODO future optimize:
  // This function is usually called in a loop. We could factor out the TIFFmalloc and TIFFfree calls.
  // Should profile to see if the repeated malloc/frees are any kind of loading bottleneck.
  if (TIFFIsTiled(tiff)) {
    tsize_t tilesize = TIFFTileSize(tiff);
    uint32 ntiles = TIFFNumberOfTiles(tiff);
    assert(ntiles == 1);
    // assuming ntiles == 1 for all IFDs
    tdata_t buf = _TIFFmalloc(tilesize);

    uint32_t i = planeIndex;

    int setdirok = TIFFSetDirectory(tiff, i);
    if (setdirok == 0) {
      LOG_ERROR << "Bad tiff directory specified: " << (i);
      _TIFFfree(buf);
      return false;
    }
    int readtileok = TIFFReadEncodedTile(tiff, 0, buf, tilesize);
    if (readtileok < 0) {
      LOG_ERROR << "Error reading tiff tile";
      _TIFFfree(buf);
      return false;
    }
    // copy buf into data.
    memcpy(dataPtr, buf, readtileok);

    _TIFFfree(buf);
  } else {
    // stripped.
    // Number of bytes in a decoded scanline
    tsize_t striplength = TIFFStripSize(tiff);
    tdata_t buf = _TIFFmalloc(striplength);

    uint32_t i = planeIndex;

    uint32_t planeindexintiff = i;
    int setdirok = TIFFSetDirectory(tiff, planeindexintiff);
    if (setdirok == 0) {
      LOG_ERROR << "Bad tiff directory specified: " << (i);
      _TIFFfree(buf);
      return false;
    }
    uint32 nstrips = TIFFNumberOfStrips(tiff);
    for (tstrip_t strip = 0; strip < nstrips; strip++) {
      int readstripok = TIFFReadEncodedStrip(tiff, strip, buf, striplength);
      if (readstripok < 0) {
        LOG_ERROR << "Error reading tiff strip";
        _TIFFfree(buf);
        return false;
      }

      // copy buf into data.
      memcpy(dataPtr, buf, readstripok);
      dataPtr += readstripok;
    }
    _TIFFfree(buf);
  }
  return true;
}

std::shared_ptr<ImageXYZC>
FileReader::loadOMETiff_4D(const std::string& filepath, bool addToCache)
{
  // check cache first of all.
  auto cached = sPreloadedImageCache.find(filepath);
  if (cached != sPreloadedImageCache.end()) {
    return cached->second;
  }

  std::shared_ptr<ImageXYZC> emptyimage;

  QElapsedTimer twhole;
  twhole.start();

  QElapsedTimer timer;
  timer.start();

  // Loads tiff file
  TIFF* tiff = TIFFOpen(filepath.c_str(), "r");
  if (!tiff) {
    QString msg = "Failed to open TIFF: '" + QString(filepath.c_str()) + "'";
    LOG_ERROR << msg.toStdString();
    // throw new Exception(NULL, msg, this, __FUNCTION__, __LINE__);
    return emptyimage;
  }

  VolumeDimensions dims;
  bool dims_ok = readTiffDimensions(tiff, filepath, dims);
  if (!dims_ok) {
    return emptyimage;
  }

  size_t planesize = dims.sizeX * dims.sizeY * dims.bitsPerPixel / 8;
  uint8_t* data = new uint8_t[planesize * dims.sizeZ * dims.sizeC];
  memset(data, 0, planesize * dims.sizeZ * dims.sizeC);
  // stash it here in case of early exit, it will be deleted
  std::unique_ptr<uint8_t[]> smartPtr(data);

  uint8_t* destptr = data;

  // now ready to read channels one by one.
  for (uint32_t channel = 0; channel < dims.sizeC; ++channel) {
    for (uint32_t slice = 0; slice < dims.sizeZ; ++slice) {
      uint32_t planeIndex = dims.getPlaneIndex(slice, channel, 0);
      destptr = data + planesize * (channel * dims.sizeZ + slice);
      if (!readTiffPlane(tiff, planeIndex, destptr)) {
        return emptyimage;
      }
    }
  }

  TIFFClose(tiff);

  LOG_DEBUG << "TIFF loaded in " << timer.elapsed() << "ms";

  // TODO: convert data to uint16_t pixels if not already.

  timer.start();
  // we can release the smartPtr because ImageXYZC will now own the raw data memory
  ImageXYZC* im = new ImageXYZC(dims.sizeX,
                                dims.sizeY,
                                dims.sizeZ,
                                dims.sizeC,
                                dims.bitsPerPixel,
                                smartPtr.release(),
                                dims.physicalSizeX,
                                dims.physicalSizeY,
                                dims.physicalSizeZ);
  LOG_DEBUG << "ImageXYZC prepared in " << timer.elapsed() << "ms";

  im->setChannelNames(dims.channelNames);

  LOG_DEBUG << "Loaded " << filepath << " in " << twhole.elapsed() << "ms";

  std::shared_ptr<ImageXYZC> sharedImage(im);
  if (addToCache) {
    sPreloadedImageCache[filepath] = sharedImage;
  }
  return sharedImage;
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
