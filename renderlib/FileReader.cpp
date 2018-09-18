#include "FileReader.h"

#include "ImageXYZC.h"
#include "Logging.h"

#include <QDomDocument>
#include <QString>
#include <QElapsedTimer>
#include <QtDebug>

#include <tiff.h>
#include <tiffio.h>

#include <map>

FileReader::FileReader()
{
}


FileReader::~FileReader()
{
}

uint32_t requireUint32Attr(QDomElement& el, const QString& attr, uint32_t defaultVal) {
	QString attrval = el.attribute(attr);
	bool ok;
	uint32_t retval = attrval.toUInt(&ok);
	if (!ok) {
		retval = defaultVal;
	}
	return retval;
}
float requireFloatAttr(QDomElement& el, const QString& attr, float defaultVal) {
	QString attrval = el.attribute(attr);
	bool ok;
	float retval = attrval.toFloat(&ok);
	if (!ok) {
		retval = defaultVal;
	}
	return retval;
}

std::shared_ptr<ImageXYZC> FileReader::loadOMETiff_4D(const std::string& filepath)
{
	QElapsedTimer timer;
	timer.start();

// Loads tiff file
  TIFF* tiff = TIFFOpen(filepath.c_str(), "r");
  if (!tiff) {
    QString msg = "Failed to open TIFF: '" + QString(filepath.c_str()) + "'";
	LOG_ERROR << msg.toStdString();
    //throw new Exception(NULL, msg, this, __FUNCTION__, __LINE__);
	return std::shared_ptr<ImageXYZC>();
  }


  char* omexmlstr = nullptr;
  // ome-xml is in ImageDescription of first IFD in the file.
  if (TIFFGetField(tiff, TIFFTAG_IMAGEDESCRIPTION, &omexmlstr) != 1) {
    QString msg = "Failed to read imagedescription of TIFF: '" + QString(filepath.c_str()) + "'";
	LOG_ERROR << msg.toStdString();

    //throw new Exception(NULL, msg, this, __FUNCTION__, __LINE__);
  }

  // Temporary variables
  uint32 width, height;
  //  tsize_t scanlength;

  // Read dimensions of image
  if (TIFFGetField(tiff, TIFFTAG_IMAGEWIDTH, &width) != 1) {
	  QString msg = "Failed to read width of TIFF: '" + QString(filepath.c_str()) + "'";
	  LOG_ERROR << msg.toStdString();
	  //throw new Exception(NULL, msg, this, __FUNCTION__, __LINE__);
  }
  if (TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &height) != 1) {
	  QString msg = "Failed to read height of TIFF: '" + QString(filepath.c_str()) + "'";
	  LOG_ERROR << msg.toStdString();
	  //throw new Exception(NULL, msg, this, __FUNCTION__, __LINE__);
  }

  uint32_t bpp = 0;
  if (TIFFGetField(tiff, TIFFTAG_BITSPERSAMPLE, &bpp) != 1) {
	  QString msg = "Failed to read bpp of TIFF: '" + QString(filepath.c_str()) + "'";
	  LOG_ERROR << msg.toStdString();
	  //throw new Exception(NULL, msg, this, __FUNCTION__, __LINE__);
  }


  uint32_t sizeT = 1;
  uint32_t sizeX = width;
  uint32_t sizeY = height;
  uint32_t sizeZ = 1;
  uint32_t sizeC = 1;
  float physicalSizeX = 1.0f;
  float physicalSizeY = 1.0f;
  float physicalSizeZ = 1.0f;
  std::vector<QString> channelNames;

  // check for plain tiff with ImageJ imagedescription:
  QString qomexmlstr(omexmlstr);
  if (qomexmlstr.startsWith("ImageJ=")) {
	  // "ImageJ=\nhyperstack=true\nimages=7900\nchannels=1\nslices=50\nframes=158"
	  QStringList sl = qomexmlstr.split('\n');
	  QRegExp reChannels("channels=(\\w+)");
	  QRegExp reSlices("slices=(\\w+)");
	  int ich = sl.indexOf(reChannels);
	  if (ich == -1) {
		  QString msg = "Failed to read number of channels of ImageJ TIFF: '" + QString(filepath.c_str()) + "'";
		  LOG_ERROR << msg.toStdString();
	  }

	  int isl = sl.indexOf(reSlices);
	  if (isl == -1) {
		  QString msg = "Failed to read number of slices of ImageJ TIFF: '" + QString(filepath.c_str()) + "'";
		  LOG_ERROR << msg.toStdString();
	  }

	  // get the n channels and n slices:
	  int pos = reChannels.indexIn(sl.at(ich));
	  if (pos > -1) {
		  QString value = reChannels.cap(1); // "189"
		  sizeC = value.toInt();
	  }
	  pos = reSlices.indexIn(sl.at(isl));
	  if (pos > -1) {
		  QString value = reSlices.cap(1); // "189"
		  sizeZ = value.toInt();
	  }
	  for (uint32_t i = 0; i < sizeC; ++i) {
		  channelNames.push_back(QString::number(i));
	  }
  }
  else {
	  // convert c to xml doc.  if this fails then we don't have an ome tif.
	  QDomDocument omexml;
	  bool ok = omexml.setContent(qomexmlstr);
	  if (!ok) {
		  QString msg = "Bad ome xml content";
		  LOG_ERROR << msg.toStdString();
		  //throw new Exception(NULL, msg, this, __FUNCTION__, __LINE__);
	  }

	  // extract some necessary info from the xml:
	  QDomElement pixelsEl = omexml.elementsByTagName("Pixels").at(0).toElement();
	  if (pixelsEl.isNull()) {
		  QString msg = "No <Pixels> element in ome xml";
		  LOG_ERROR << msg.toStdString();
	  }

	  // skipping "complex", "double-complex", and "bit".
	  std::map<std::string, uint32_t> mapPixelTypeBPP = {
		  { "uint8", 8 },
		  { "uint16", 16 },
		  { "uint32", 32 },
		  { "int8", 8 },
		  { "int16", 16 },
		  { "int32", 32 },
		  { "float", 32 },
		  { "double", 64 }
	  };

	  QString pixelType = pixelsEl.attribute("PixelType", "uint16").toLower();
	  bpp = mapPixelTypeBPP[pixelType.toStdString()];
	  sizeX = requireUint32Attr(pixelsEl, "SizeX", 0);
	  sizeY = requireUint32Attr(pixelsEl, "SizeY", 0);
	  sizeZ = requireUint32Attr(pixelsEl, "SizeZ", 0);
	  sizeC = requireUint32Attr(pixelsEl, "SizeC", 0);
	  sizeT = requireUint32Attr(pixelsEl, "SizeT", 0);
	  // one of : "XYZCT", "XYZTC","XYCTZ","XYCZT","XYTCZ","XYTZC"
	  QString dimensionOrder = pixelsEl.attribute("DimensionOrder", "XYCZT");
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
			  channelNames.push_back(chname);
		  }
		  else if (!chid.isEmpty()) {
			  channelNames.push_back(chid);
		  }
		  else {
			  channelNames.push_back(QString("%1").arg(i));
		  }
	  }

  }

  assert(sizeX == width);
  assert(sizeY == height);


	// allocate the destination buffer!!!!
  assert(sizeC >= 1);
  assert(sizeX >= 1);
  assert(sizeY >= 1);
  assert(sizeZ >= 1);
  size_t planesize = sizeX*sizeY*bpp / 8;
  uint8_t* data = new uint8_t[planesize*sizeZ*sizeC];
  memset(data, 0, planesize*sizeZ*sizeC);



  uint8_t* destptr = data;

  if (TIFFIsTiled(tiff)) {
	  tsize_t tilesize = TIFFTileSize(tiff);
	  uint32 ntiles = TIFFNumberOfTiles(tiff);
	  assert(ntiles == 1);
	  // assuming ntiles == 1 for all IFDs
	  tdata_t buf = _TIFFmalloc(tilesize);
	  for (uint32_t i = 0; i < sizeC; ++i) {
		  for (uint32_t j = 0; j < sizeZ; ++j) {
			  int setdirok = TIFFSetDirectory(tiff, j + i*sizeZ);
			  if (setdirok == 0) {
				  LOG_ERROR << "Bad tiff directory specified: " << (j + i*sizeZ);
			  }
			  int readtileok = TIFFReadEncodedTile(tiff, 0, buf, tilesize);
			  if (readtileok < 0) {
				  LOG_ERROR << "Error reading tiff tile";
			  }
			  // copy buf into data.
			  memcpy(destptr, buf, tilesize);
			  destptr += tilesize;
		  }
	  }
	  _TIFFfree(buf);
  }
  else {
	  // stripped.
	  // Number of bytes in a decoded scanline
	  tsize_t striplength = TIFFStripSize(tiff);
	  tdata_t buf = _TIFFmalloc(striplength);
	  for (uint32_t i = 0; i < sizeZ; ++i) {
		  for (uint32_t j = 0; j < sizeC; ++j) {
			  uint32_t planeindexintiff = j + i*sizeC;
			  int setdirok = TIFFSetDirectory(tiff, planeindexintiff);
			  if (setdirok == 0) {
				  LOG_ERROR << "Bad tiff directory specified: " << (j + i*sizeC);
			  }
			  // ensure channels are coalesced (transposing from xycz to xyzc)
			  uint32_t planeindexinbuffer = i + j*sizeZ;
			  destptr = data + (planesize*(planeindexinbuffer));
			  uint32 nstrips = TIFFNumberOfStrips(tiff);
			  for (tstrip_t strip = 0; strip < nstrips; strip++)
			  {
				  int readstripok = TIFFReadEncodedStrip(tiff, strip, buf, striplength);
				  if (readstripok < 0) {
					  LOG_ERROR << "Error reading tiff strip";
				  }

				  // copy buf into data.
				  memcpy(destptr, buf, striplength);
				  destptr += striplength;
			  }

		  }
	  }
	  _TIFFfree(buf);

  }

  TIFFClose(tiff);	

  LOG_DEBUG << "TIFF loaded in " << timer.elapsed() << "ms";

  timer.start();
  ImageXYZC* im = new ImageXYZC(sizeX, sizeY, sizeZ, sizeC, uint32_t(bpp), data, physicalSizeX, physicalSizeY, physicalSizeZ);
  LOG_DEBUG << "ImageXYZC prepared in " << timer.elapsed() << "ms";
  
  im->setChannelNames(channelNames);

  return std::shared_ptr<ImageXYZC>(im);
}
