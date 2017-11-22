#include "RenderGL.h"

#include "glad/glad.h"

#include "gl/v33/V33Image3D.h"
#include "Camera.h"
#include "Scene.h"

#include <iostream>

RenderGL::RenderGL(std::shared_ptr<ImageXYZC>  img, CScene* scene)
	:image3d(nullptr),
	_scene(scene),
	_img(img)
{
}


RenderGL::~RenderGL()
{
	delete image3d;
}

void RenderGL::initialize(uint32_t w, uint32_t h)
{
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	//glEnable(GL_MULTISAMPLE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	image3d = new Image3Dv33(_img);

	GLint max_combined_texture_image_units;
	glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &max_combined_texture_image_units);
	std::cout << "Texture unit count: " << max_combined_texture_image_units << std::endl;

	image3d->create();

	// Size viewport
	resize(w,h);
}

void RenderGL::render(const Camera& camera)
{
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Render image
	image3d->setC(_scene->_channel);
	image3d->render(camera);
}

void RenderGL::resize(uint32_t w, uint32_t h)
{
	glViewport(0, 0, w, h);
}
