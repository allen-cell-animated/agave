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

class GLImageShader2DnoLut;
class ImageXYZC;
class Image3Dv33;
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

	virtual CStatus* getStatusInterface() { return &_status; }

	Image3Dv33* getImage() const { return nullptr; };
	RenderSettings& getRenderSettings() { return *_renderSettings; }

	// just draw into my own fbo.
	void doRender(const CCamera& camera);
	// draw my fbo texture into the current render target
	void drawImage();
private:
	RenderSettings* _renderSettings;

	RenderParams _renderParams;
	Scene* _scene;

	void initQuad();
	void initFB(uint32_t w, uint32_t h);
	void initVolumeTextureCUDA();

	ImageCuda _imgCuda;

	/// The vertex array.
	GLuint vertices;  // vao
	/// The image vertices.
	GLuint image_vertices;  // buffer
	/// The image texture coordinates.
	GLuint image_texcoords; // buffer
	/// The image elements.
	GLuint image_elements;  // buffer
	size_t num_image_elements;
	/// The identifier of the texture owned and used by this object.
	unsigned int textureid;
	GLImageShader2DnoLut *image_shader;


	// the rgba8 buffer for display
	cudaGraphicsResource* _cudaTex;
	cudaSurfaceObject_t _cudaGLSurfaceObject;
	GLuint _fbtex;

	// the rgbaf32 buffer for rendering
	float* _cudaF32Buffer;
	// the rgbaf32 accumulation buffer that holds the progressively rendered image
	float* _cudaF32AccumBuffer;

	// screen size auxiliary buffers for rendering 
	unsigned int* _randomSeeds1;
	unsigned int* _randomSeeds2;

	int _w, _h;

	CTiming _timingRender, _timingBlur, _timingPostProcess, _timingDenoise;
	CStatus _status;

	void FillCudaLighting(Scene* pScene, CudaLighting& cl);
    void FillCudaCamera(const CCamera* pCamera, CudaCamera& c);
	void cleanUpFB();

};

