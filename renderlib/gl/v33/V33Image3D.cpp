
#include "glad/glad.h"
#include "gl/v33/V33Image3D.h"
#include "gl/Util.h"
#include "ImageXYZC.h"

#include <ome/files/PixelBuffer.h>
#include <ome/files/VariantPixelBuffer.h>

#include <iostream>


namespace
{

	class TextureProperties
	{
	public:
		GLenum internal_format;
		GLenum external_format;
		GLint external_type;
		bool make_normal;
		GLint min_filter;
		GLint mag_filter;
		ome::files::dimension_size_type w;
		ome::files::dimension_size_type h;

		TextureProperties(const ome::files::FormatReader& reader,
			ome::files::dimension_size_type series) :
			internal_format(GL_R8),
			external_format(GL_RED),
			external_type(GL_UNSIGNED_BYTE),
			make_normal(false),
			min_filter(GL_LINEAR),
			mag_filter(GL_LINEAR),
			w(0),
			h(0)
		{
			ome::files::dimension_size_type oldseries = reader.getSeries();
			reader.setSeries(series);
			ome::xml::model::enums::PixelType pixeltype = reader.getPixelType();
			reader.setSeries(oldseries);

			w = reader.getSizeX();
			h = reader.getSizeY();

			switch (pixeltype)
			{
			case ::ome::xml::model::enums::PixelType::INT8:
				internal_format = GL_R8;
				external_type = GL_BYTE;
				break;
			case ::ome::xml::model::enums::PixelType::INT16:
				internal_format = GL_R16;
				external_type = GL_SHORT;
				break;
			case ::ome::xml::model::enums::PixelType::INT32:
				internal_format = GL_R16;
				external_type = GL_INT;
				make_normal = true;
				break;
			case ::ome::xml::model::enums::PixelType::UINT8:
				internal_format = GL_R8;
				external_type = GL_UNSIGNED_BYTE;
				break;
			case ::ome::xml::model::enums::PixelType::UINT16:
				internal_format = GL_R16;
				external_type = GL_UNSIGNED_SHORT;
				break;
			case ::ome::xml::model::enums::PixelType::UINT32:
				internal_format = GL_R16;
				external_type = GL_UNSIGNED_INT;
				make_normal = true;
				break;
			case ::ome::xml::model::enums::PixelType::FLOAT:
				internal_format = GL_R32F;
				if (!GL_ARB_texture_float)
					internal_format = GL_R16;
				external_type = GL_FLOAT;
				break;
			case ::ome::xml::model::enums::PixelType::DOUBLE:
				internal_format = GL_R32F;
				if (!GL_ARB_texture_float)
					internal_format = GL_R16;
				external_type = GL_DOUBLE;
				break;
			case ::ome::xml::model::enums::PixelType::BIT:
				internal_format = GL_R8;
				external_type = GL_UNSIGNED_BYTE;
				make_normal = true;
				min_filter = GL_NEAREST_MIPMAP_LINEAR;
				mag_filter = GL_NEAREST;
				break;
			case ::ome::xml::model::enums::PixelType::COMPLEXFLOAT:
				internal_format = GL_RG32F;
				if (!GL_ARB_texture_float)
					internal_format = GL_RG16;
				external_type = GL_FLOAT;
				external_format = GL_RG;
			case ::ome::xml::model::enums::PixelType::COMPLEXDOUBLE:
				internal_format = GL_RG32F;
				if (!GL_ARB_texture_float)
					internal_format = GL_RG16;
				external_type = GL_DOUBLE;
				external_format = GL_RG;
				break;
			}
		}
	};
}

Image3Dv33::Image3Dv33(std::shared_ptr<ome::files::FormatReader>  reader,
	std::shared_ptr<ImageXYZC>  img,
	ome::files::dimension_size_type                    series):
	vertices(0),
	image_vertices(0),
	image_elements(0),
	num_image_elements(0),
	textureid(0),
	lutid(0),
	texmin(0.0f),
	texmax(1.0f),
	//texcorr(1.0f),
	reader(reader),
	series(series),
	_img(img),
	image3d_shader(new GLBasicVolumeShader()),
	_c(0)
{
}

Image3Dv33::~Image3Dv33()
{
	delete image3d_shader;
}

void Image3Dv33::create()
{
	TextureProperties tprop(*reader, series);

	ome::files::dimension_size_type oldseries = reader->getSeries();
	reader->setSeries(series);
	ome::files::dimension_size_type sizeX = reader->getSizeX();
	ome::files::dimension_size_type sizeY = reader->getSizeY();
	ome::files::dimension_size_type sizeZ = reader->getSizeZ();
	setSize(glm::vec2(-(sizeX / 2.0f), sizeX / 2.0f),
		glm::vec2(-(sizeY / 2.0f), sizeY / 2.0f));
	ome::files::dimension_size_type rbpp = reader->getBitsPerPixel();
	ome::files::dimension_size_type bpp = ome::files::bitsPerPixel(reader->getPixelType());
	//texcorr[0] = texcorr[1] = texcorr[2] = (1 << (bpp - rbpp));
	reader->setSeries(oldseries);

	// Create image texture.
	glGenTextures(1, &textureid);
	glBindTexture(GL_TEXTURE_3D, textureid);
	check_gl("Bind texture");
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, tprop.min_filter);
	check_gl("Set texture min filter");
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, tprop.mag_filter);
	check_gl("Set texture mag filter");
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	check_gl("Set texture wrap s");
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	check_gl("Set texture wrap t");
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	check_gl("Set texture wrap r");

	// assuming 16-bit data!
	Channelu16* ch = _img->channel(_c);
	texmin = float(ch->_min) / (65535.0f);
	texmax = float(ch->_max) / (65535.0f);
/*
	uint16_t test[] = {
		1024, 128, 256, 512, 
		1024, 2048, 1024, 512
	};
	sizeX = 2;
	sizeY = 2;
	sizeZ = 2;
	texmin = 0.0;
	texmax = 2048.0 / (255.0*64.0);
	*/
	glTexImage3D(GL_TEXTURE_3D,         // target
		0,                     // level, 0 = base, no minimap,
		tprop.internal_format, // internal format
		(GLsizei)sizeX,                 // width
		(GLsizei)sizeY,                 // height
		(GLsizei)sizeZ,
		0,                     // border
		tprop.external_format, // external format
		tprop.external_type,   // external type
		_img->ptr(_c));
		//0);                    // no image data at this point
	check_gl("Texture create");
	glGenerateMipmap(GL_TEXTURE_3D);

	// Create LUT texture.
	glGenTextures(1, &lutid);
	glBindTexture(GL_TEXTURE_1D_ARRAY, lutid);
	check_gl("Bind texture");
	glTexParameteri(GL_TEXTURE_1D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	check_gl("Set texture min filter");
	glTexParameteri(GL_TEXTURE_1D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	check_gl("Set texture mag filter");
	glTexParameteri(GL_TEXTURE_1D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	check_gl("Set texture wrap s");

	// HiLo
	uint8_t lut[256][3];
	for (uint16_t i = 0; i < 256; ++i)
		for (uint16_t j = 0; j < 3; ++j)
		{
			lut[i][j] = (uint8_t)i;
		}
	lut[0][0] = 0;
	lut[0][2] = 0;
	lut[0][2] = 255;
	lut[255][0] = 255;
	lut[255][1] = 0;
	lut[255][2] = 0;

	glTexImage2D(GL_TEXTURE_1D_ARRAY, // target
		0,                   // level, 0 = base, no minimap,
		GL_RGB8,             // internal format
		256,                 // width
		1,                   // height
		0,                   // border
		GL_RGB,              // external format
		GL_UNSIGNED_BYTE,    // external type
		lut);                // LUT data
	check_gl("Texture create");
}

void
Image3Dv33::render(const Camera& camera)
{
	image3d_shader->bind();

	image3d_shader->dataRangeMin = texmin;
	image3d_shader->dataRangeMax = texmax;
	image3d_shader->GAMMA_MIN = 0.0;
	image3d_shader->GAMMA_MAX = 1.0;
	image3d_shader->GAMMA_SCALE = 1.0;
	image3d_shader->BRIGHTNESS = 1.0;
	image3d_shader->DENSITY = 0.0820849986238988f;
	image3d_shader->maskAlpha = 1.0;
	image3d_shader->BREAK_STEPS = 512;
	image3d_shader->AABB_CLIP_MIN = glm::vec3(-0.5,-0.5,-0.5);
	image3d_shader->AABB_CLIP_MAX = glm::vec3(0.5,0.5,0.5);
	image3d_shader->setShadingUniforms();

	image3d_shader->setTransformUniforms(camera, glm::mat4(1.0f));

	glActiveTexture(GL_TEXTURE0);
	check_gl("Activate texture");
	glBindTexture(GL_TEXTURE_3D, textureid);
	check_gl("Bind texture");
	image3d_shader->setTexture(0);

	glActiveTexture(GL_TEXTURE1);
	check_gl("Activate texture");
	glBindTexture(GL_TEXTURE_1D_ARRAY, lutid);
	check_gl("Bind texture");
	image3d_shader->setLUT(1);

	glBindVertexArray(vertices);

	image3d_shader->enableCoords();
	image3d_shader->setCoords(image_vertices, 0, 3);

	// Push each element to the vertex shader
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, image_elements);
	glDrawElements(GL_TRIANGLES, (GLsizei)num_image_elements, GL_UNSIGNED_SHORT, 0);
	check_gl("Image3Dv33 draw elements");


	image3d_shader->disableCoords();
	glBindVertexArray(0);

	image3d_shader->release();
}

void
Image3Dv33::setSize(const glm::vec2& xlim,
	const glm::vec2& ylim)
{
	const std::array<GLfloat, 3*4*2> cube_vertices
	{
		// front
		-0.5, -0.5,  0.5,
		0.5, -0.5,  0.5,
		0.5,  0.5,  0.5,
		-0.5,  0.5,  0.5,
		// back
		-0.5, -0.5, -0.5,
		0.5, -0.5, -0.5,
		0.5,  0.5, -0.5,
		-0.5,  0.5, -0.5, 
	};

	if (vertices == 0) {
		glGenVertexArrays(1, &vertices);
	}
	glBindVertexArray(vertices);

	if (image_vertices == 0) {
		glGenBuffers(1, &image_vertices);
	}
	glBindBuffer(GL_ARRAY_BUFFER, image_vertices);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * cube_vertices.size(), cube_vertices.data(), GL_STATIC_DRAW);

	//note every face of the cube is on a single line
	std::array<GLushort, 36> cube_indices = {
		// front
		0, 1, 2,
		2, 3, 0,
		// top
		1, 5, 6,
		6, 2, 1,
		// back
		7, 6, 5,
		5, 4, 7,
		// bottom
		4, 0, 3,
		3, 7, 4,
		// left
		4, 5, 1,
		1, 0, 4,
		// right
		3, 2, 6,
		6, 7, 3,
	};

	if (image_elements == 0) {
		glGenBuffers(1, &image_elements);
	}
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, image_elements);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort) * cube_indices.size(), cube_indices.data(), GL_STATIC_DRAW);
	num_image_elements = cube_indices.size();
}

unsigned int
Image3Dv33::texture()
{
	return textureid;
}

unsigned int
Image3Dv33::lut()
{
	return lutid;
}

void 
Image3Dv33::setPlane(int plane, int z, int c)
{
	if (c != _c)
	{
		_c = c;
		// only update C here!!
		TextureProperties tprop(*reader, series);
		ome::files::dimension_size_type sizeX = reader->getSizeX();
		ome::files::dimension_size_type sizeY = reader->getSizeY();
		ome::files::dimension_size_type sizeZ = reader->getSizeZ();

		// assuming 16-bit data!
		Channelu16* ch = _img->channel(c);
		texmin = float(ch->_min) / (65535.0f);
		texmax = float(ch->_max) / (65535.0f);

		glBindTexture(GL_TEXTURE_3D, textureid);
		glTexImage3D(GL_TEXTURE_3D,         // target
			0,                     // level, 0 = base, no minimap,
			tprop.internal_format, // internal format
			(GLsizei)sizeX,                 // width
			(GLsizei)sizeY,                 // height
			(GLsizei)sizeZ,
			0,                     // border
			tprop.external_format, // external format
			tprop.external_type,   // external type
			_img->ptr(c));
		glGenerateMipmap(GL_TEXTURE_3D);
	}
}

