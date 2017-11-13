#include "FileReader.h"

#include <ome/files/FormatReader.h>
#include <ome/files/MetadataTools.h>
#include <ome/files/PixelBuffer.h>
#include <ome/files/VariantPixelBuffer.h>
#include <ome/files/in/OMETIFFReader.h>
#include <ome/xml/meta/MetadataStore.h>
#include <ome/xml/meta/OMEXMLMetadata.h>

#include "ImageXYZC.h"

FileReader::FileReader(const std::string& omeSchemaDir)
{
	std::string env = "OME_XML_SCHEMADIR=" + omeSchemaDir;
	_putenv(env.c_str());
}


FileReader::~FileReader()
{
}

std::shared_ptr<ome::files::FormatReader> FileReader::open(std::string& filepath) {
	
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

std::shared_ptr<ImageXYZC> FileReader::openToImage(std::string& filepath) {
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
