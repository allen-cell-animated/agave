#include "RenderGLOptix.h"

#include "glad/glad.h"
#include "glm.h"

#include "gl/Util.h"
#include "ImageXYZC.h"
#include "Logging.h"

#include <array>

RenderGLOptix::RenderGLOptix(RenderSettings* rs)
	: _renderSettings(rs),
	_w(0),
	_h(0),
	_scene(nullptr),
	_gpuBytes(0)
{
}


RenderGLOptix::~RenderGLOptix()
{
}

void RenderGLOptix::initialize(uint32_t w, uint32_t h)
{

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	check_gl("init gl state");

	// Size viewport
	resize(w,h);
}

void RenderGLOptix::doRender(const CCamera& camera) {
	if (!_scene || !_scene->_volume) {
		return;
	}
	
}

void RenderGLOptix::render(const CCamera& camera)
{
	// draw to _fbtex
	doRender(camera);

	// put _fbtex to main render target
	drawImage();
}

void RenderGLOptix::drawImage() {
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

}


void RenderGLOptix::resize(uint32_t w, uint32_t h)
{
	//w = 8; h = 8;
	glViewport(0, 0, w, h);
	if ((_w == w) && (_h == h)) {
		return;
	}

	LOG_DEBUG << "Resized window to " << w << " x " << h;
}

RenderParams& RenderGLOptix::renderParams() {
	return _renderParams;
}
Scene* RenderGLOptix::scene() {
	return _scene;
}
void RenderGLOptix::setScene(Scene* s) {
	_scene = s;
}

size_t RenderGLOptix::getGpuBytes() {
	return 0;
}
