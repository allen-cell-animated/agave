#include "RenderGL2d.h"

#include "glad/glad.h"

#include "gl/v33/V33Image2D.h"
#include "Camera.h"

#include <iostream>

RenderGL2d::RenderGL2d(std::shared_ptr<ImageXYZC>  img)
	:image(nullptr),
	_img(img)
{
}


RenderGL2d::~RenderGL2d()
{
	delete image;
}

void RenderGL2d::initialize(uint32_t w, uint32_t h)
{
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	//glEnable(GL_MULTISAMPLE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	image = new Image2Dv33(_img);

	GLint max_combined_texture_image_units;
	glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &max_combined_texture_image_units);
	std::cout << "Texture unit count: " << max_combined_texture_image_units << std::endl;

	image->create();

	// Size viewport
	resize(w,h);
}

void RenderGL2d::render(const Camera& camera)
{
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Render image
	glm::mat4 mvp = camera.mvp();
	image->render(mvp);
}

void RenderGL2d::resize(uint32_t w, uint32_t h)
{
	glViewport(0, 0, w, h);
}
