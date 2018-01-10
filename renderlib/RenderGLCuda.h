#pragma once
#include "IRenderWindow.h"

#include "AppScene.h"
#include "Scene.h"

#include "glad/include/glad/glad.h"
#include "CudaUtilities.h"
#include "ImageXyzcCuda.h"

#include <memory>

class GLImageShader2DnoLut;
class ImageXYZC;
class Image3Dv33;

class RenderGLCuda :
	public IRenderWindow
{
public:
	RenderGLCuda(std::shared_ptr<ImageXYZC>  img, CScene* scene);
	virtual ~RenderGLCuda();

	virtual void initialize(uint32_t w, uint32_t h);
	virtual void render(const Camera& camera);
	virtual void resize(uint32_t w, uint32_t h);
	virtual void cleanUpResources();
	virtual RenderParams& renderParams();
	virtual Scene& scene();


	void setImage(std::shared_ptr<ImageXYZC> img);
	Image3Dv33* getImage() const { return nullptr; };
	CScene& getScene() { return *_renderSettings; }

	// just draw into my own fbo.
	void doRender();
	// draw my fbo texture into the current render target
	void drawImage();
private:
	CScene* _renderSettings;

	RenderParams _renderParams;
	Scene _appScene;

	void initSceneLighting();

	void initQuad();
	void initFB(uint32_t w, uint32_t h);
	void initVolumeTextureCUDA();



	std::shared_ptr<ImageXYZC>  _img;
	int _currentChannel;
	void initSceneFromImg();

	ImageCuda _imgCuda;
	CScene* _deviceScene;

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
	GLuint _fbtex;

	// the rgbaf32 buffer for rendering
	float* _cudaF32Buffer;
	// the rgbaf32 accumulation buffer that holds the progressively rendered image
	float* _cudaF32AccumBuffer;

	// screen size auxiliary buffers for rendering 
	unsigned int* _randomSeeds1;
	unsigned int* _randomSeeds2;

	int _w, _h;

};

