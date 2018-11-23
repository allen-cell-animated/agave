
#include "glad/glad.h"
#include "gl/v33/V33Image3D.h"
#include "gl/Util.h"
#include "Logging.h"
#include "ImageXYZC.h"
#include "RenderSettings.h"

#include <QElapsedTimer>

#include <array>
#include <iostream>

Image3Dv33::Image3Dv33(std::shared_ptr<ImageXYZC>  img) :
	vertices(0),
	image_vertices(0),
	image_elements(0),
	num_image_elements(0),
	_textureid(0),
	_lutid(0),
	texmin(0.0f),
	texmax(1.0f),
	//texcorr(1.0f),
	_img(img),
	image3d_shader(new GLBasicVolumeShader()),
	_c(0),
	_fusedrgbvolume(nullptr)
{
}

Image3Dv33::~Image3Dv33()
{
	delete[] _fusedrgbvolume;

	glDeleteTextures(1, &_textureid);
	glDeleteTextures(1, &_lutid);
	delete image3d_shader;
}

void Image3Dv33::create()
{
	_fusedrgbvolume = new uint8_t[3 * _img->sizeX() * _img->sizeY() * _img->sizeZ()];
	// destroy old
	glDeleteTextures(1, &_textureid);
	// Create image texture.
	glGenTextures(1, &_textureid);

	setSize(glm::vec2(-(_img->sizeX() / 2.0f), _img->sizeX() / 2.0f),
		glm::vec2(-(_img->sizeY() / 2.0f), _img->sizeY() / 2.0f));

	//setC(_c, true);
	//prepareTexture();
	// Create LUT texture.
	glGenTextures(1, &_lutid);
	glBindTexture(GL_TEXTURE_1D_ARRAY, _lutid);
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
Image3Dv33::render(const CCamera& camera, const Scene* scene, const RenderSettings* renderSettings)
{
	image3d_shader->bind();

	image3d_shader->dataRangeMin = texmin;
	image3d_shader->dataRangeMax = texmax;
	image3d_shader->GAMMA_MIN = 0.0;
	image3d_shader->GAMMA_MAX = 1.0;
	image3d_shader->GAMMA_SCALE = 1.3657f;
	image3d_shader->BRIGHTNESS = (1.0f-camera.m_Film.m_Exposure) + 1.0f;
	//_renderSettings.m_RenderSettings.m_DensityScale
	image3d_shader->DENSITY = renderSettings->m_RenderSettings.m_DensityScale / 100.0;
	image3d_shader->maskAlpha = 1.0;
	image3d_shader->BREAK_STEPS = 512;
	// axis aligned clip planes in object space
	image3d_shader->AABB_CLIP_MIN = scene->m_roi.GetMinP() - glm::vec3(0.5, 0.5, 0.5);
	image3d_shader->AABB_CLIP_MAX = scene->m_roi.GetMaxP() - glm::vec3(0.5, 0.5, 0.5);
	image3d_shader->setShadingUniforms();

	// move the box to match where the camera is pointed
	// transform the box from -0.5..0.5 to 0..physicalsize
	glm::vec3 dims(_img->sizeX()*_img->physicalSizeX(),
		_img->sizeY()*_img->physicalSizeY(),
		_img->sizeZ()*_img->physicalSizeZ());
	float maxd = std::max(dims.x, std::max(dims.y, dims.z));
	glm::vec3 scales(dims.x / maxd, dims.y / maxd, dims.z / maxd);
	// it helps to imagine these transforming the space in reverse order
	// (first translate by 0.5, and then scale)
	glm::mat4 mm = glm::scale(glm::mat4(1.0f), scales);
	mm = glm::translate(mm, glm::vec3(0.5, 0.5, 0.5));

	image3d_shader->setTransformUniforms(camera, mm);

	glActiveTexture(GL_TEXTURE0);
	check_gl("Activate texture");
	glBindTexture(GL_TEXTURE_3D, _textureid);
	check_gl("Bind texture");
	image3d_shader->setTexture(0);

	glActiveTexture(GL_TEXTURE1);
	check_gl("Activate texture");
	glBindTexture(GL_TEXTURE_1D_ARRAY, _lutid);
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
	return _textureid;
}

unsigned int
Image3Dv33::lut()
{
	return _lutid;
}

void 
Image3Dv33::setC(int c, bool force)
{
	if (force || (c != _c))
	{
		_c = c;
		// only update C here!!

		// assuming 16-bit data!
		Channelu16* ch = _img->channel(c);
		texmin = float(ch->m_min) / (65535.0f);
		texmax = float(ch->m_max) / (65535.0f);

		GLenum internal_format = GL_R16;
		GLenum external_type = GL_UNSIGNED_SHORT;
		GLenum external_format = GL_RED;

		// pixel data is tightly packed
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		glBindTexture(GL_TEXTURE_3D, _textureid);
		glTexImage3D(GL_TEXTURE_3D,         // target
			0,                     // level, 0 = base, no minimap,
			internal_format, // internal format
			(GLsizei)_img->sizeX(),                 // width
			(GLsizei)_img->sizeY(),                 // height
			(GLsizei)_img->sizeZ(),
			0,                     // border
			external_format, // external format
			external_type,   // external type
			ch->m_ptr);
		check_gl("Volume Texture create");
		glGenerateMipmap(GL_TEXTURE_3D);
	}
}

void Image3Dv33::prepareTexture(Scene& s) {
	QElapsedTimer timer;
	timer.start();

	std::vector<glm::vec3> colors;
	for (int i = 0; i < MAX_CPU_CHANNELS; ++i) {
		if (s.m_material.enabled[i]) {
			colors.push_back(glm::vec3(s.m_material.diffuse[i * 3],
				s.m_material.diffuse[i * 3 + 1],
				s.m_material.diffuse[i * 3 + 2]) * s.m_material.opacity[i]);
		}
		else {
			colors.push_back(glm::vec3(0, 0, 0));
		}
	}

	_img->fuse(colors, &_fusedrgbvolume, nullptr);
	
	LOG_DEBUG << "fuse operation: " << timer.elapsed() << "ms";
	timer.start();

	// destroy old
	//glDeleteTextures(1, &_textureid);

	// Create image texture.
	//glGenTextures(1, &_textureid);
	glBindTexture(GL_TEXTURE_3D, _textureid);
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
	
	GLenum internal_format = GL_RGBA8;
	GLenum external_type = GL_UNSIGNED_BYTE;
	GLenum external_format = GL_RGB;

	// pixel data is tightly packed
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

	glTexImage3D(GL_TEXTURE_3D,         // target
		0,                     // level, 0 = base, no minimap,
		internal_format, // internal format
		(GLsizei)_img->sizeX(),                 // width
		(GLsizei)_img->sizeY(),                 // height
		(GLsizei)_img->sizeZ(),
		0,                     // border
		external_format, // external format
		external_type,   // external type
		_fusedrgbvolume);
	check_gl("Volume Texture create");
//	glGenerateMipmap(GL_TEXTURE_3D);

	LOG_DEBUG << "prepare fused 3d rgb texture in " << timer.elapsed() << "ms";
}

