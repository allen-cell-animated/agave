#include "RenderGL.h"

#include "glad/glad.h"

#include "gl/v33/V33Image3D.h"
#include "Camera.h"
#include "ImageXYZC.h"
#include "Logging.h"
#include "Scene.h"

#include <iostream>

RenderGL::RenderGL(CScene* scene)
	:image3d(nullptr),
	_w(0),
	_h(0),
	_scene(scene)
{
}

void RenderGL::setImage(std::shared_ptr<ImageXYZC> img)
{
	_appScene._volume = img;

	delete image3d;
	image3d = new Image3Dv33(img);
	image3d->create();

	_scene->initSceneFromImg(img->sizeX(), img->sizeY(), img->sizeZ(),
		img->physicalSizeX(), img->physicalSizeY(), img->physicalSizeZ());

	// we have set up everything there is to do before rendering
	_timer.start();
	_status.SetRenderBegin();

}

RenderGL::~RenderGL()
{
	delete image3d;
}

void RenderGL::initialize(uint32_t w, uint32_t h)
{
	GLint max_combined_texture_image_units;
	glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &max_combined_texture_image_units);
	LOG_DEBUG << "GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS: " << max_combined_texture_image_units;

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	//glEnable(GL_MULTISAMPLE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	if (_appScene._volume) {
		image3d = new Image3Dv33(_appScene._volume);
		image3d->create();
	}

	// Size viewport
	resize(w,h);
}

void RenderGL::render(const Camera& camera)
{
	if (!_appScene._volume) {
		return;
	}
	if (!image3d) {
		image3d = new Image3Dv33(_appScene._volume);
		image3d->create();
	}

	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if (!image3d) {
		return;
	}


	if (_scene->m_DirtyFlags.HasFlag(RenderParamsDirty | TransferFunctionDirty | VolumeDataDirty))
	{
		image3d->prepareTexture(_appScene);
	}

	// At this point, all dirty flags should have been taken care of, since the flags in the original scene are now cleared
	_scene->m_DirtyFlags.ClearAllFlags();

	_scene->m_Camera.m_Film.m_Resolution.SetResX(_w);
	_scene->m_Camera.m_Film.m_Resolution.SetResY(_h);

	_scene->m_Camera.Update();

	// Render image
	image3d->render(_scene->m_Camera);

	_timingRender.AddDuration((float)_timer.elapsed());
	_status.SetStatisticChanged("Performance", "Render Image", QString::number(_timingRender.m_FilteredDuration, 'f', 2), "ms.");
	_timer.start();

}

void RenderGL::resize(uint32_t w, uint32_t h)
{
	_w = w;
	_h = h;
	glViewport(0, 0, w, h);
}

RenderParams& RenderGL::renderParams() {
	return _renderParams;
}
Scene& RenderGL::scene() {
	return _appScene;
}
