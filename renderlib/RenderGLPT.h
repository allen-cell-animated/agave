#pragma once
#include "IRenderWindow.h"

#include "AppScene.h"
#include "RenderSettings.h"

#include "glad/include/glad/glad.h"
#include "CudaUtilities.h"
#include "ImageXyzcCuda.h"
#include "Status.h"
#include "Timing.h"

#include <memory>

class FSQ;
class ImageXYZC;
class Image3Dv33;
class RectImage2D;
struct CudaLighting;
struct CudaCamera;

class RenderGLPT :
	public IRenderWindow
{
public:
	RenderGLPT(RenderSettings* rs);
	virtual ~RenderGLPT();

	virtual void initialize(uint32_t w, uint32_t h);
	virtual void render(const CCamera& camera);
	virtual void resize(uint32_t w, uint32_t h);
	virtual void cleanUpResources();
	virtual RenderParams& renderParams();
	virtual Scene* scene();
	virtual void setScene(Scene* s);

	virtual CStatus* getStatusInterface() { return &_status; }

	Image3Dv33* getImage() const { return nullptr; };
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

	void initFB(uint32_t w, uint32_t h);
	void initVolumeTextureCUDA();
	void cleanUpFB();

	ImageGL _imgCuda;

	RectImage2D* _imagequad;

	// the rgba8 buffer for display
	GLuint _fbtex;
    GLuint _fb;

	// the rgbaf32 buffer for rendering
	GLuint _glF32Buffer;
    GLuint _fbF32;
    FSQ* _renderBufferShader;

	// the rgbaf32 accumulation buffer that holds the progressively rendered image
    GLuint _glF32AccumBuffer;
    GLuint _glF32AccumBuffer2; // for ping ponging
    GLuint _fbF32Accum;
    FSQ* _accumBufferShader;

	// screen size auxiliary buffers for rendering 
	unsigned int* _randomSeeds1;
	unsigned int* _randomSeeds2;

	int _w, _h;

	CTiming _timingRender, _timingBlur, _timingPostProcess, _timingDenoise;
	CStatus _status;

	size_t _gpuBytes;

	void FillCudaLighting(Scene* pScene, CudaLighting& cl);
    void FillCudaCamera(const CCamera* pCamera, CudaCamera& c);

};

