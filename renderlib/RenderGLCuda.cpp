#include "RenderGLCuda.h"

#include "glad/glad.h"
#include "glm.h"

#include "gl/Util.h"
#include "gl/v33/V33Image3D.h"
#include "glsl/v330/V330GLImageShader2DnoLut.h"
#include "Camera.h"
#include "ImageXYZC.h"
#include "Logging.h"
#include "cudarndr/RenderThread.h"

#include <cuda_runtime.h>

#include "Core.cuh"
#include "Lighting.cuh"

#include <array>

RenderGLCuda::RenderGLCuda(std::shared_ptr<ImageXYZC>  img, CScene* scene)
	:_img(img),
	_cudaF32Buffer(nullptr),
	_cudaF32AccumBuffer(nullptr),
	_cudaTex(nullptr),
	_fbtex(0),
	vertices(0),
	image_vertices(0),
	image_texcoords(0),
	image_elements(0),
	_randomSeeds1(nullptr),
	_randomSeeds2(nullptr),
	_currentChannel(0),
	_renderSettings(scene),
	_w(0),
	_h(0)
{
	initSceneLighting();
	initSceneFromImg();
	_renderSettings->m_Camera.SetViewMode(ViewModeFront);
}


RenderGLCuda::~RenderGLCuda()
{
}

void RenderGLCuda::initSceneLighting() {
	CLight BackgroundLight;

	BackgroundLight.m_T = 1;
	float inten = 1.0f;

	float topr = 1.0f;
	float topg = 0.0f;
	float topb = 0.0f;
	float midr = 1.0f;
	float midg = 1.0f;
	float midb = 1.0f;
	float botr = 0.0f;
	float botg = 0.0f;
	float botb = 1.0f;

	BackgroundLight.m_ColorTop = inten * CColorRgbHdr(topr, topg, topb);
	BackgroundLight.m_ColorMiddle = inten * CColorRgbHdr(midr, midg, midb);
	BackgroundLight.m_ColorBottom = inten * CColorRgbHdr(botr, botg, botb);

	BackgroundLight.Update(_renderSettings->m_BoundingBox);

	_renderSettings->m_Lighting.AddLight(BackgroundLight);

	CLight AreaLight;

	AreaLight.m_T = 0;
	AreaLight.m_Theta = 0.0f / RAD_F;  // numerator is degrees
	AreaLight.m_Phi = 0.0f / RAD_F;
	AreaLight.m_Width = 1.0f;
	AreaLight.m_Height = 1.0f;
	AreaLight.m_Distance = 10.0f;
	AreaLight.m_Color = 100.0f * CColorRgbHdr(1.0f, 1.0f, 1.0f);

	AreaLight.Update(_renderSettings->m_BoundingBox);

	_renderSettings->m_Lighting.AddLight(AreaLight);
}

void RenderGLCuda::initSceneFromImg()
{
	if (!_img) {
		return;
	}

	_renderSettings->m_Resolution.SetResX(_img->sizeX());
	_renderSettings->m_Resolution.SetResY(_img->sizeY());
	_renderSettings->m_Resolution.SetResZ(_img->sizeZ());
	_renderSettings->m_Spacing.x = _img->physicalSizeX();
	_renderSettings->m_Spacing.y = _img->physicalSizeY();
	_renderSettings->m_Spacing.z = _img->physicalSizeZ();

	//Log("Spacing: " + FormatSize(gScene.m_Spacing, 2), "grid");

	// Compute physical size
	const Vec3f PhysicalSize(Vec3f(
		_renderSettings->m_Spacing.x * (float)_renderSettings->m_Resolution.GetResX(), 
		_renderSettings->m_Spacing.y * (float)_renderSettings->m_Resolution.GetResY(), 
		_renderSettings->m_Spacing.z * (float)_renderSettings->m_Resolution.GetResZ()
	));

	// Compute the volume's bounding box
	_renderSettings->m_BoundingBox.m_MinP = Vec3f(0.0f);
	_renderSettings->m_BoundingBox.m_MaxP = PhysicalSize / PhysicalSize.Max();

	_renderSettings->m_Camera.m_SceneBoundingBox = _renderSettings->m_BoundingBox;


	for (int i = 0; i < _renderSettings->m_Lighting.m_NoLights; ++i) {
		_renderSettings->m_Lighting.m_Lights[i].Update(_renderSettings->m_BoundingBox);
	}
}

void RenderGLCuda::initQuad()
{
	check_gl("begin initQuad ");
	// setup geometry
	glm::vec2 xlim(-1.0, 1.0);
	glm::vec2 ylim(-1.0, 1.0);
	const std::array<GLfloat, 8> square_vertices
	{
		xlim[0], ylim[0],
		xlim[1], ylim[0],
		xlim[1], ylim[1],
		xlim[0], ylim[1]
	};

	if (vertices == 0) {
		glGenVertexArrays(1, &vertices);
	}
	glBindVertexArray(vertices);
	check_gl("create and bind verts");

	if (image_vertices == 0) {
		glGenBuffers(1, &image_vertices);
	}
	glBindBuffer(GL_ARRAY_BUFFER, image_vertices);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * square_vertices.size(), square_vertices.data(), GL_STATIC_DRAW);
	check_gl("init vtx coord data");

	glm::vec2 texxlim(0.0, 1.0);
	glm::vec2 texylim(0.0, 1.0);
	std::array<GLfloat, 8> square_texcoords
	{
		texxlim[0], texylim[0],
		texxlim[1], texylim[0],
		texxlim[1], texylim[1],
		texxlim[0], texylim[1]
	};

	if (image_texcoords == 0) {
		glGenBuffers(1, &image_texcoords);
	}
	glBindBuffer(GL_ARRAY_BUFFER, image_texcoords);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * square_texcoords.size(), square_texcoords.data(), GL_STATIC_DRAW);
	check_gl("init texcoord data");

	std::array<GLushort, 6> square_elements
	{
		// front
		0,  1,  2,
		2,  3,  0
	};

	if (image_elements == 0) {
		glGenBuffers(1, &image_elements);
	}
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, image_elements);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort) * square_elements.size(), square_elements.data(), GL_STATIC_DRAW);
	num_image_elements = square_elements.size();
	check_gl("init element data");

	glBindVertexArray(0);
	check_gl("unbind vtx array");
}

void RenderGLCuda::initFB(uint32_t w, uint32_t h)
{
	_w = w;
	_h = h;

	if (_cudaF32Buffer) {
		HandleCudaError(cudaFree(_cudaF32Buffer));
		_cudaF32Buffer = nullptr;
	}
	if (_cudaF32AccumBuffer) {
		HandleCudaError(cudaFree(_cudaF32AccumBuffer));
		_cudaF32AccumBuffer = nullptr;
	}
	if (_randomSeeds1) {
		HandleCudaError(cudaFree(_randomSeeds1));
		_randomSeeds1 = nullptr;
	}
	if (_randomSeeds2) {
		HandleCudaError(cudaFree(_randomSeeds2));
		_randomSeeds2 = nullptr;
	}
	if (_cudaTex) {
		HandleCudaError(cudaGraphicsUnregisterResource(_cudaTex));
	}
	if (_fbtex) {
		glDeleteTextures(1, &_fbtex);
		check_gl("Destroy fb texture");
	}
	
	HandleCudaError(cudaMalloc((void**)&_cudaF32Buffer, w*h * 4 * sizeof(float)));
	HandleCudaError(cudaMemset(_cudaF32Buffer, 0, w*h * 4 * sizeof(float)));
	HandleCudaError(cudaMalloc((void**)&_cudaF32AccumBuffer, w*h * 4 * sizeof(float)));
	HandleCudaError(cudaMemset(_cudaF32AccumBuffer, 0, w*h * 4 * sizeof(float)));

	{
		unsigned int* pSeeds = (unsigned int*)malloc(w*h * sizeof(unsigned int));

		HandleCudaError(cudaMalloc((void**)&_randomSeeds1, w*h * sizeof(unsigned int)));
		memset(pSeeds, 0, w*h * sizeof(unsigned int));
		for (unsigned int i = 0; i < w*h; i++)
			pSeeds[i] = rand();
		HandleCudaError(cudaMemcpy(_randomSeeds1, pSeeds, w*h * sizeof(unsigned int), cudaMemcpyHostToDevice));


		HandleCudaError(cudaMalloc((void**)&_randomSeeds2, w*h * sizeof(unsigned int)));
		memset(pSeeds, 0, w*h * sizeof(unsigned int));
		for (unsigned int i = 0; i < w*h; i++)
			pSeeds[i] = rand();
		HandleCudaError(cudaMemcpy(_randomSeeds2, pSeeds, w*h * sizeof(unsigned int), cudaMemcpyHostToDevice));

		free(pSeeds);
	}

	glGenTextures(1, &_fbtex);
	check_gl("Gen fb texture id");
	glBindTexture(GL_TEXTURE_2D, _fbtex);
	check_gl("Bind fb texture");
	//glTextureStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, w, h);


	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	check_gl("Create fb texture");
	// this is required in order to "complete" the texture object for mipmapless shader access.
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	// unbind the texture before doing cuda stuff.
	glBindTexture(GL_TEXTURE_2D, 0);
	
	// use gl interop to let cuda write to this tex.
	HandleCudaError(cudaGraphicsGLRegisterImage(&_cudaTex, _fbtex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));
}

void RenderGLCuda::initVolumeTextureCUDA() {
	if (!_img) {
		return;
	}
	ImageCuda cimg;
	cimg.allocGpu(_img.get());
	_imgCuda = cimg;
}

void RenderGLCuda::setImage(std::shared_ptr<ImageXYZC> img) {
	// free the gpu resources of the old image.
	_imgCuda.deallocGpu();

	_img = img;
	initVolumeTextureCUDA();
	initSceneFromImg();
}
void RenderGLCuda::initialize(uint32_t w, uint32_t h)
{
	initQuad();
	check_gl("init quad");

	image_shader = new GLImageShader2DnoLut();
	check_gl("init simple image shader");

	initVolumeTextureCUDA();

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	check_gl("init gl state");

	// Size viewport
	resize(w,h);
}

void RenderGLCuda::doRender() {
	if (!_img) {
		return;
	}

	_currentChannel = _renderSettings->_channel;

	// Resizing the image canvas requires special attention
	if (_renderSettings->m_DirtyFlags.HasFlag(FilmResolutionDirty))
	{
#if 0
		// Allocate host image buffer, this thread will blit it's frames to this buffer
		free(m_pRenderImage);
		m_pRenderImage = NULL;

		m_pRenderImage = (CColorRgbLdr*)malloc(_renderSettings->m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorRgbLdr));

		if (m_pRenderImage)
			memset(m_pRenderImage, 0, _renderSettings->m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorRgbLdr));

		gStatus.SetStatisticChanged("Host Memory", "LDR Frame Buffer", QString::number(3 * _renderSettings->m_Camera.m_Film.m_Resolution.GetNoElements() * sizeof(CColorRgbLdr) / MB, 'f', 2), "MB");
#endif
		_renderSettings->SetNoIterations(0);

		//Log("Render canvas resized to: " + QString::number(SceneCopy.m_Camera.m_Film.m_Resolution.GetResX()) + " x " + QString::number(SceneCopy.m_Camera.m_Film.m_Resolution.GetResY()) + " pixels", "application-resize");
	}

	// Restart the rendering when when the camera, lights and render params are dirty
	if (_renderSettings->m_DirtyFlags.HasFlag(CameraDirty | LightsDirty | RenderParamsDirty | TransferFunctionDirty))
	{
		_imgCuda.updateLutGpu(_currentChannel, _img.get());

		//		ResetRenderCanvasView();

		// Reset no. iterations
		_renderSettings->SetNoIterations(0);
	}

	// At this point, all dirty flags should have been taken care of, since the flags in the original scene are now cleared
	_renderSettings->m_DirtyFlags.ClearAllFlags();

	// TODO: update only when w and h change!
	_renderSettings->m_Camera.m_Film.m_Resolution.SetResX(_w);
	_renderSettings->m_Camera.m_Film.m_Resolution.SetResY(_h);

	// TODO: update only for channel changes!
	//	_renderSettings->m_IntensityRange.SetMin(0.0f);
	_renderSettings->m_IntensityRange.SetMin((float)(_img->channel(_currentChannel)->_min));
	_renderSettings->m_IntensityRange.SetMax((float)(_img->channel(_currentChannel)->_max));

	_renderSettings->m_Camera.Update();

	_renderSettings->m_GradientDelta = 1.0f / (float)_renderSettings->m_Resolution.GetMax();

	BindConstants(_renderSettings);
	// Render image
	//RayMarchVolume(_cudaF32Buffer, _volumeTex, _volumeGradientTex, _renderSettings, _w, _h, 2.0f, 20.0f, glm::value_ptr(m), _channelMin, _channelMax);
	cudaFB theCudaFB = {
		_cudaF32Buffer,
		_cudaF32AccumBuffer,
		_randomSeeds1,
		_randomSeeds2
	};

	// single channel
	int NC = _img->sizeC();
	cudaVolume theCudaVolume(3);
	for (int i = 0; i < min(NC, 8); ++i) {
		theCudaVolume.volumeTexture[i] = _imgCuda._channels[i]._volumeTexture;
		theCudaVolume.gradientVolumeTexture[i] = _imgCuda._channels[i]._volumeGradientTexture;
		theCudaVolume.lutTexture[i] = _imgCuda._channels[i]._volumeLutTexture;
		theCudaVolume.intensityMax[i] = _img->channel(i)->_max;
		switch (i) {
		case 0:
			theCudaVolume.diffuse[i * 3 + 0] = 1.0;
			theCudaVolume.diffuse[i * 3 + 1] = 0.0;
			theCudaVolume.diffuse[i * 3 + 2] = 1.0;
			break;
		case 1:
			theCudaVolume.diffuse[i * 3 + 0] = 1.0;
			theCudaVolume.diffuse[i * 3 + 1] = 1.0;
			theCudaVolume.diffuse[i * 3 + 2] = 1.0;
			break;
		case 2:
			theCudaVolume.diffuse[i * 3 + 0] = 0.0;
			theCudaVolume.diffuse[i * 3 + 1] = 1.0;
			theCudaVolume.diffuse[i * 3 + 2] = 1.0;
			break;
		case 3:
			theCudaVolume.diffuse[i * 3 + 0] = 1.0;
			theCudaVolume.diffuse[i * 3 + 1] = 1.0;
			theCudaVolume.diffuse[i * 3 + 2] = 0.0;
			break;
		}
	}

	CTiming ri, bi, ppi, di;


	Render(0, *_renderSettings,
		theCudaFB,
		theCudaVolume,
		ri, bi, ppi, di);
	//LOG_DEBUG << "RETURN FROM RENDER";

	// Tonemap into opengl display buffer

	// do cuda with cudaSurfaceObj
	HandleCudaError(cudaGraphicsMapResources(1, &_cudaTex));
	{
		cudaArray_t ca;
		HandleCudaError(cudaGraphicsSubResourceGetMappedArray(&ca, _cudaTex, 0, 0));
		cudaResourceDesc desc;
		memset(&desc, 0, sizeof(desc));
		desc.resType = cudaResourceTypeArray;
		desc.res.array.array = ca;
		cudaSurfaceObject_t theCudaSurfaceObject;
		HandleCudaError(cudaCreateSurfaceObject(&theCudaSurfaceObject, &desc));

		// set the lerpC here because the Render call is incrementing the number of iterations.

		_renderSettings->m_DenoiseParams.SetWindowRadius(3.0f);
		//_renderSettings->m_DenoiseParams.m_LerpC = 0.33f * (max((float)_renderSettings->GetNoIterations(), 1.0f) * 1.0f);//1.0f - powf(1.0f / (float)gScene.GetNoIterations(), 15.0f);//1.0f - expf(-0.01f * (float)gScene.GetNoIterations());
		_renderSettings->m_DenoiseParams.m_LerpC = 0.33f * (max((float)_renderSettings->GetNoIterations(), 1.0f) * 0.035f);//1.0f - powf(1.0f / (float)gScene.GetNoIterations(), 15.0f);//1.0f - expf(-0.01f * (float)gScene.GetNoIterations());

		if (_renderSettings->m_DenoiseParams.m_Enabled && _renderSettings->m_DenoiseParams.m_LerpC > 0.0f && _renderSettings->m_DenoiseParams.m_LerpC < 1.0f)
		{
			Denoise(_cudaF32AccumBuffer, theCudaSurfaceObject, _w, _h);
		}
		else
		{
			ToneMap(_cudaF32AccumBuffer, theCudaSurfaceObject, _w, _h);
		}
		HandleCudaError(cudaDestroySurfaceObject(theCudaSurfaceObject));
	}
	HandleCudaError(cudaGraphicsUnmapResources(1, &_cudaTex));

	HandleCudaError(cudaStreamSynchronize(0));
}

void RenderGLCuda::render(const Camera& camera)
{
	// draw to _fbtex
	doRender();

	// put _fbtex to main render target
	drawImage();
}

void RenderGLCuda::drawImage() {
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// draw quad using the tex that cudaTex was mapped to

	image_shader->bind();
	check_gl("Bind shader");

	image_shader->setModelViewProjection(glm::mat4(1.0));

	glActiveTexture(GL_TEXTURE0);
	check_gl("Activate texture");
	glBindTexture(GL_TEXTURE_2D, _fbtex);
	check_gl("Bind texture");
	image_shader->setTexture(0);

	glBindVertexArray(vertices);
	check_gl("bind vtx buf");

	image_shader->enableCoords();
	image_shader->setCoords(image_vertices, 0, 2);

	image_shader->enableTexCoords();
	image_shader->setTexCoords(image_texcoords, 0, 2);

	// Push each element to the vertex shader
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, image_elements);
	check_gl("bind element buf");
	glDrawElements(GL_TRIANGLES, (GLsizei)num_image_elements, GL_UNSIGNED_SHORT, 0);
	check_gl("Image2D draw elements");

	image_shader->disableCoords();
	image_shader->disableTexCoords();
	glBindVertexArray(0);
	glBindTexture(GL_TEXTURE_2D, 0);

	image_shader->release();
}


void RenderGLCuda::resize(uint32_t w, uint32_t h)
{
	glViewport(0, 0, w, h);
	if ((_w == w) && (_h == h)) {
		return;
	}

	initFB(w, h);
}

void RenderGLCuda::cleanUpResources() {
	glDeleteVertexArrays(1, &vertices);
	glDeleteBuffers(1, &image_vertices);
	glDeleteBuffers(1, &image_texcoords);
	glDeleteBuffers(1, &image_elements);

	glDeleteTextures(1, &_fbtex);
	if (_cudaF32Buffer) {
		HandleCudaError(cudaFree(_cudaF32Buffer));
		_cudaF32Buffer = nullptr;
	}
	if (_cudaF32AccumBuffer) {
		HandleCudaError(cudaFree(_cudaF32AccumBuffer));
		_cudaF32AccumBuffer = nullptr;
	}
	if (_randomSeeds1) {
		HandleCudaError(cudaFree(_randomSeeds1));
		_randomSeeds1 = nullptr;
	}
	if (_randomSeeds2) {
		HandleCudaError(cudaFree(_randomSeeds2));
		_randomSeeds2 = nullptr;
	}
	if (_cudaTex) {
		HandleCudaError(cudaGraphicsUnregisterResource(_cudaTex));
	}
}

RenderParams& RenderGLCuda::renderParams() {
	return _renderParams;
}
Scene& RenderGLCuda::scene() {
	return _appScene;
}
