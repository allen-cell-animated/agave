#include "FileReader.h"

#include "ImageXYZC.h"
#include "Logging.h"

#include <QDomDocument>
#include <QString>

#include <tiff.h>
#include <tiffio.h>

#include <map>

#ifdef WIN32
static int setenv(const char *name, const char *value, int overwrite)
{
    int errcode = 0;
    if(!overwrite) {
        size_t envsize = 0;
        errcode = getenv_s(&envsize, NULL, 0, name);
        if(errcode || envsize) return errcode;
    }
    return _putenv_s(name, value);
}
#endif

FileReader::FileReader(const std::string& omeSchemaDir)
{
	//std::string env = "OME_XML_SCHEMADIR=" + omeSchemaDir;
	setenv("OME_XML_SCHEMADIR", omeSchemaDir.c_str(), 1);
}


FileReader::~FileReader()
{
}
#if 0
std::shared_ptr<ome::files::FormatReader> FileReader::open(const std::string& filepath) {
	
	std::shared_ptr<ome::files::FormatReader> reader(std::make_shared<ome::files::in::OMETIFFReader>());
//	try {
		reader->setId(filepath);
//	}
//	catch (ome::files::FormatException& e) {
//	}
	return reader;
}

struct CopyBufferVisitor : public boost::static_visitor<>
{
	CopyBufferVisitor(uint8_t* dest, uint32_t size) { _dest = dest; _size = size; }
	uint8_t* _dest;
	uint32_t _size;

	ome::files::PixelBufferBase::storage_order_type
		gl_order(const ome::files::PixelBufferBase::storage_order_type& order)
	{
		ome::files::PixelBufferBase::storage_order_type ret(order);
		// This makes the assumption that the order is SXY or XYS, and
		// switches XYS to SXY if needed.
		if (order.ordering(0) != ome::files::DIM_SUBCHANNEL)
		{
			ome::files::PixelBufferBase::size_type ordering[ome::files::PixelBufferBase::dimensions];
			bool ascending[ome::files::PixelBufferBase::dimensions] = { true, true, true, true, true, true, true, true, true };
			for (boost::detail::multi_array::size_type d = 0; d < ome::files::PixelBufferBase::dimensions; ++d)
			{
				ordering[d] = order.ordering(d);
				ascending[d] = order.ascending(d);

				ome::files::PixelBufferBase::size_type xo = ordering[0];
				ome::files::PixelBufferBase::size_type yo = ordering[1];
				ome::files::PixelBufferBase::size_type so = ordering[2];
				bool xa = ascending[0];
				bool ya = ascending[1];
				bool sa = ascending[2];

				ordering[0] = so;
				ordering[1] = xo;
				ordering[2] = yo;
				ascending[0] = sa;
				ascending[1] = xa;
				ascending[2] = ya;

				ret = ome::files::PixelBufferBase::storage_order_type(ordering, ascending);
			}
		}
		return ret;
	}

	template<typename T>
	void
		operator() (const T& v)
	{
		typedef typename T::element_type::value_type value_type;

		T src_buffer(v);
		const ome::files::PixelBufferBase::storage_order_type& orig_order(v->storage_order());
		ome::files::PixelBufferBase::storage_order_type new_order(gl_order(orig_order));

		if (!(new_order == orig_order))
		{
			// Reorder as interleaved.
			const ome::files::PixelBufferBase::size_type *shape = v->shape();

			T gl_buf(new typename T::element_type(boost::extents[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][shape[6]][shape[7]][shape[8]],
				v->pixelType(),
				v->endianType(),
				new_order));
			*gl_buf = *v;
			src_buffer = gl_buf;
		}


		memcpy(_dest, v->data(), _size);
	}

	template <typename T>
	typename boost::enable_if_c<
		boost::is_complex<T>::value, void
	>::type
		operator() (const std::shared_ptr<ome::files::PixelBuffer<T>>& /* v */)
	{
		/// @todo Conversion from complex.
	}

};

std::shared_ptr<ImageXYZC> FileReader::openToImage(const std::string& filepath) {
//	boost::filesystem::path p(filepath);
//	std::shared_ptr<ome::xml::meta::OMEXMLMetadata> filemeta(ome::files::createOMEXMLMetadata(p));
// assumes unit is the same for all three (microns?)
//	float ppx = (float)(filemeta->getPixelsPhysicalSizeX(0).getValue());
//	float ppy = (float)(filemeta->getPixelsPhysicalSizeY(0).getValue());
//	float ppz = (float)(filemeta->getPixelsPhysicalSizeZ(0).getValue());
	float ppx = 0.065f;
	float ppy = 0.065f;
	float ppz = 0.290f;


	std::shared_ptr<ome::files::FormatReader> reader(std::make_shared<ome::files::in::OMETIFFReader>());
	reader->setId(filepath);

	reader->setSeries(0);


	//std::shared_ptr<ome::xml::meta::OMEXMLMetadata> filemeta(ome::files::createOMEXMLMetadata(*reader));

	ome::files::dimension_size_type x = reader->getSizeX();
	ome::files::dimension_size_type y = reader->getSizeY();
	ome::files::dimension_size_type z = reader->getSizeZ();
	ome::files::dimension_size_type c = reader->getSizeC();
	
	assert(reader->getSizeT() == 1);
	assert(c >= 1);
	assert(reader->getRGBChannelCount(0) == 1);

	ome::files::pixel_size_type bpp = reader->getBitsPerPixel();
	
	ome::files::dimension_size_type imageCount = reader->getImageCount();
	assert(imageCount == z*c);

	size_t planesize = x*y*bpp / 8;
	uint8_t* data = new uint8_t[planesize*z*c];

	ome::files::VariantPixelBuffer buf;

//	for ()
//	dimension_size_type index = reader->getIndex(z, c, t);
	std::array<ome::files::dimension_size_type, 3> zct;
	for (uint32_t plane = 0; plane < imageCount; ++plane) {

		reader->openBytes(plane, buf);
		zct = reader->getZCTCoords(plane);

		// advance by bytes into the buffer...
		CopyBufferVisitor v(data + ((zct[0]+zct[1]*z)*planesize), uint32_t(planesize));
		boost::apply_visitor(v, buf.vbuffer());
	}

	// image takes ownership of the data ptr.
	return std::shared_ptr<ImageXYZC>(new ImageXYZC(uint32_t(x), uint32_t(y), uint32_t(z), uint32_t(c), uint32_t(bpp), data, ppx, ppy, ppz));
}
#endif
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
// Loads tiff file
  TIFF* tiff = TIFFOpen(filepath.c_str(), "r");
  if (!tiff) {
    QString msg = "Failed to open TIFF: '" + QString(filepath.c_str()) + "'";
	LOG_ERROR << msg.toStdString();
    //throw new Exception(NULL, msg, this, __FUNCTION__, __LINE__);
  }


  char* omexmlstr = nullptr;
  // ome-xml is in ImageDescription of first IFD in the file.
  if (TIFFGetField(tiff, TIFFTAG_IMAGEDESCRIPTION, &omexmlstr) != 1) {
    QString msg = "Failed to read width of TIFF: '" + QString(filepath.c_str()) + "'";
	LOG_ERROR << msg.toStdString();

    //throw new Exception(NULL, msg, this, __FUNCTION__, __LINE__);
  }
  // convert c to xml doc.  if this fails then we don't have an ome tif.
  QDomDocument omexml;
  bool ok = omexml.setContent(QString(omexmlstr));
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
  uint32_t bpp = mapPixelTypeBPP[pixelType.toStdString()];
  uint32_t sizeX = requireUint32Attr(pixelsEl, "SizeX", 0);
  uint32_t sizeY = requireUint32Attr(pixelsEl, "SizeY", 0);
  uint32_t sizeZ = requireUint32Attr(pixelsEl, "SizeZ", 0);
  uint32_t sizeC = requireUint32Attr(pixelsEl, "SizeC", 0);
  uint32_t sizeT = requireUint32Attr(pixelsEl, "SizeT", 0);
  // one of : "XYZCT", "XYZTC","XYCTZ","XYCZT","XYTCZ","XYTZC"
  QString dimensionOrder = pixelsEl.attribute("DimensionOrder", "XYCZT");
  float physicalSizeX = requireFloatAttr(pixelsEl, "PhysicalSizeX", 0.0f);
  float physicalSizeY = requireFloatAttr(pixelsEl, "PhysicalSizeY", 0.0f);
  float physicalSizeZ = requireFloatAttr(pixelsEl, "PhysicalSizeZ", 0.0f);
  QString physicalSizeXunit = pixelsEl.attribute("PhysicalSizeXUnit", "");
  QString physicalSizeYunit = pixelsEl.attribute("PhysicalSizeYUnit", "");
  QString physicalSizeZunit = pixelsEl.attribute("PhysicalSizeZUnit", "");


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

  assert(sizeX == width);
  assert(sizeY == height);
#if 0
  uint32 tilewidth, tileheight;
  if (TIFFGetField(tiff, TIFFTAG_TILEWIDTH, &tilewidth) != 1) {
	  QString msg = "Failed to read tilewidth of TIFF: '" + QString(filepath.c_str()) + "'";
	  LOG_ERROR << msg.toStdString();
  }
  if (TIFFGetField(tiff, TIFFTAG_TILELENGTH, &tileheight) != 1) {
	  QString msg = "Failed to read tileheight of TIFF: '" + QString(filepath.c_str()) + "'";
	  LOG_ERROR << msg.toStdString();
  }
#endif

	// allocate the destination buffer!!!!
  assert(sizeC >= 1);
  assert(sizeX >= 1);
  assert(sizeY >= 1);
  assert(sizeZ >= 1);
  size_t planesize = sizeX*sizeY*bpp / 8;
  uint8_t* data = new uint8_t[planesize*sizeZ*sizeC];



  // Number of bytes in a decoded scanline
  tsize_t scanlength = TIFFScanlineSize(tiff);
  tsize_t tilesize = TIFFTileSize(tiff);
  uint32 ntiles = TIFFNumberOfTiles(tiff);
  assert(ntiles == 1);


// assuming ntiles == 1 for all IFDs
  tdata_t buf = _TIFFmalloc(TIFFTileSize(tiff));

  uint8_t* destptr = data;
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

  TIFFClose(tiff);	


  return std::shared_ptr<ImageXYZC>(new ImageXYZC(sizeX, sizeY, sizeZ, sizeC, uint32_t(bpp), data, physicalSizeX, physicalSizeY, physicalSizeZ));
}
