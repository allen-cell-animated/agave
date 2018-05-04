#pragma once
#include "IRenderWindow.h"

#include "AppScene.h"
#include "RenderSettings.h"

#include "glad/include/glad/glad.h"
#include "Status.h"
#include "Timing.h"

#include <memory>

class ImageXYZC;

class RenderGLOptix :
	public IRenderWindow
{
public:
	RenderGLOptix(RenderSettings* rs);
	virtual ~RenderGLOptix();

	virtual void initialize(uint32_t w, uint32_t h);
	virtual void render(const CCamera& camera);
	virtual void resize(uint32_t w, uint32_t h);
	virtual void cleanUpResources();
	virtual RenderParams& renderParams();
	virtual Scene* scene();
	virtual void setScene(Scene* s);

	virtual CStatus* getStatusInterface() { return &_status; }

	RenderSettings& getRenderSettings() { return *_renderSettings; }

	// just draw into my own fbo.
	void doRender(const CCamera& camera);
	// draw my fbo texture into the current render target
	void drawImage();

	size_t getGpuBytes();
private:
	RenderSettings* _renderSettings;
	RenderParams _renderParams;
	Scene* _scene;

	int _w, _h;

	CStatus _status;

	size_t _gpuBytes;

};

