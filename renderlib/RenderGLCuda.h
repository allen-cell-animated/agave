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

class ImageXYZC;
class Image3Dv33;
class RectImage2D;
struct CudaLighting;
struct CudaCamera;

class RenderGLCuda :
	public IRenderWindow
{
public:
	RenderGLCuda(RenderSettings* rs);
	virtual ~RenderGLCuda();

	virtual void initialize(uint32_t w, uint32_t h);
	virtual void render(const CCamera& camera);
	virtual void resize(uint32_t w, uint32_t h);
	virtual void cleanUpResources();
	virtual RenderParams& renderParams();
	virtual Scene* scene();
	virtual void setScene(Scene* s);

	virtual CStatus* getStatusInterface() { return &m_status; }

	Image3Dv33* getImage() const { return nullptr; };
	RenderSettings& getRenderSettings() { return *m_renderSettings; }

	// just draw into my own fbo.
	void doRender(const CCamera& camera);
	// draw my fbo texture into the current render target
	void drawImage();

	size_t getGpuBytes();
private:
	RenderSettings* m_renderSettings;
	RenderParams m_renderParams;
	Scene* m_scene;

	void initFB(uint32_t w, uint32_t h);
	void initVolumeTextureCUDA();
	void cleanUpFB();

	std::shared_ptr<ImageCuda> m_imgCuda;

	RectImage2D* m_imagequad;

	// the rgba8 buffer for display
	cudaGraphicsResource* m_cudaTex;
	cudaSurfaceObject_t m_cudaGLSurfaceObject;
	GLuint m_fbtex;

	// the rgbaf32 buffer for rendering
	float* m_cudaF32Buffer;
	// the rgbaf32 accumulation buffer that holds the progressively rendered image
	float* m_cudaF32AccumBuffer;

	// screen size auxiliary buffers for rendering 
	unsigned int* m_randomSeeds1;
	unsigned int* m_randomSeeds2;

	int m_w, m_h;

	CTiming m_timingRender, m_timingBlur, m_timingPostProcess, m_timingDenoise;
	CStatus m_status;

	size_t m_gpuBytes;

	void FillCudaLighting(Scene* pScene, CudaLighting& cl);
    void FillCudaCamera(const CCamera* pCamera, CudaCamera& c);

};

