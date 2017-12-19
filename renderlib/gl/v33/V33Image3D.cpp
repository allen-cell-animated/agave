
#include "glad/glad.h"
#include "gl/v33/V33Image3D.h"
#include "gl/Util.h"
#include "ImageXYZC.h"

#include <array>
#include <iostream>

Image3Dv33::Image3Dv33(std::shared_ptr<ImageXYZC>  img):
	vertices(0),
	image_vertices(0),
	image_elements(0),
	num_image_elements(0),
	textureid(0),
	lutid(0),
	texmin(0.0f),
	texmax(1.0f),
	//texcorr(1.0f),
	_img(img),
	image3d_shader(new GLBasicVolumeShader()),
	_c(0)
{
}

Image3Dv33::~Image3Dv33()
{
	glDeleteTextures(1, &textureid);
	glDeleteTextures(1, &lutid);
	delete image3d_shader;
}

void Image3Dv33::create()
{

	setSize(glm::vec2(-(_img->sizeX() / 2.0f), _img->sizeX() / 2.0f),
		glm::vec2(-(_img->sizeY() / 2.0f), _img->sizeY() / 2.0f));

	// Create image texture.
	glGenTextures(1, &textureid);
	glBindTexture(GL_TEXTURE_3D, textureid);
	check_gl("Bind texture");
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	check_gl("Set texture min filter");
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
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
	GLenum internal_format = GL_R16;
	GLenum external_type = GL_UNSIGNED_SHORT;
	GLenum external_format = GL_RED;

	glTexImage3D(GL_TEXTURE_3D,         // target
		0,                     // level, 0 = base, no minimap,
		internal_format, // internal format
		(GLsizei)_img->sizeX(),                 // width
		(GLsizei)_img->sizeY(),                 // height
		(GLsizei)_img->sizeZ(),
		0,                     // border
		external_format, // external format
		external_type,   // external type
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
Image3Dv33::setC(int c)
{
	if (c != _c)
	{
		_c = c;
		// only update C here!!

		// assuming 16-bit data!
		Channelu16* ch = _img->channel(c);
		texmin = float(ch->_min) / (65535.0f);
		texmax = float(ch->_max) / (65535.0f);

		GLenum internal_format = GL_R16;
		GLenum external_type = GL_UNSIGNED_SHORT;
		GLenum external_format = GL_RED;

		glBindTexture(GL_TEXTURE_3D, textureid);
		glTexImage3D(GL_TEXTURE_3D,         // target
			0,                     // level, 0 = base, no minimap,
			internal_format, // internal format
			(GLsizei)_img->sizeX(),                 // width
			(GLsizei)_img->sizeY(),                 // height
			(GLsizei)_img->sizeZ(),
			0,                     // border
			external_format, // external format
			external_type,   // external type
			_img->ptr(c));
		glGenerateMipmap(GL_TEXTURE_3D);

	}
}

