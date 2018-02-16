#include "RenderGLCuda.h"

#include "glad/glad.h"
#include "glm.h"

#include "gl/Util.h"
#include "gl/v33/V33Image3D.h"
#include "glsl/v330/V330GLImageShader2DnoLut.h"
#include "ImageXYZC.h"
#include "Logging.h"
#include "cudarndr/RenderThread.h"

#include "Core.cuh"
#include "Lighting2.cuh"
#include "Camera2.cuh"

#include <array>

RenderGLCuda::RenderGLCuda(RenderSettings* rs)
	:_cudaF32Buffer(nullptr),
	_cudaF32AccumBuffer(nullptr),
	_cudaTex(nullptr),
	_cudaGLSurfaceObject(0),
	_fbtex(0),
	_quadVertexArray(0),
	_quadVertices(0),
	_quadTexcoords(0),
	_quadIndices(0),
	_randomSeeds1(nullptr),
	_randomSeeds2(nullptr),
	_renderSettings(rs),
	_w(0),
	_h(0),
	_scene(nullptr),
	_gpuBytes(0)
{
}


RenderGLCuda::~RenderGLCuda()
{
}

void gVec3ToFloat3(const glm::vec3* src, float3* dest) {
	dest->x = src->x;
	dest->y = src->y;
	dest->z = src->z;
}

void RenderGLCuda::FillCudaCamera(const CCamera* pCamera, CudaCamera& c) {
    gVec3ToFloat3(&pCamera->m_From, &c.m_From);
    gVec3ToFloat3(&pCamera->m_N, &c.m_N);
    gVec3ToFloat3(&pCamera->m_U, &c.m_U);
    gVec3ToFloat3(&pCamera->m_V, &c.m_V);
    c.m_ApertureSize = pCamera->m_Aperture.m_Size;
    c.m_FocalDistance = pCamera->m_Focus.m_FocalDistance;
    c.m_InvScreen[0] = pCamera->m_Film.m_InvScreen.x;
    c.m_InvScreen[1] = pCamera->m_Film.m_InvScreen.y;
    c.m_Screen[0][0] = pCamera->m_Film.m_Screen[0][0];
    c.m_Screen[1][0] = pCamera->m_Film.m_Screen[1][0];
    c.m_Screen[0][1] = pCamera->m_Film.m_Screen[0][1];
    c.m_Screen[1][1] = pCamera->m_Film.m_Screen[1][1];
}

void RenderGLCuda::FillCudaLighting(Scene* pScene, CudaLighting& cl) {
	cl.m_NoLights = pScene->_lighting.m_NoLights;
	for (int i = 0; i < min(cl.m_NoLights, MAX_CUDA_LIGHTS); ++i) {
		cl.m_Lights[i].m_Theta = pScene->_lighting.m_Lights[i].m_Theta;
		cl.m_Lights[i].m_Phi = pScene->_lighting.m_Lights[i].m_Phi;
		cl.m_Lights[i].m_Width = pScene->_lighting.m_Lights[i].m_Width;
		cl.m_Lights[i].m_InvWidth = pScene->_lighting.m_Lights[i].m_InvWidth;
		cl.m_Lights[i].m_HalfWidth = pScene->_lighting.m_Lights[i].m_HalfWidth;
		cl.m_Lights[i].m_InvHalfWidth = pScene->_lighting.m_Lights[i].m_InvHalfWidth;
		cl.m_Lights[i].m_Height = pScene->_lighting.m_Lights[i].m_Height;
		cl.m_Lights[i].m_InvHeight = pScene->_lighting.m_Lights[i].m_InvHeight;
		cl.m_Lights[i].m_HalfHeight = pScene->_lighting.m_Lights[i].m_HalfHeight;
		cl.m_Lights[i].m_InvHalfHeight = pScene->_lighting.m_Lights[i].m_InvHalfHeight;
		cl.m_Lights[i].m_Distance = pScene->_lighting.m_Lights[i].m_Distance;
		cl.m_Lights[i].m_SkyRadius = pScene->_lighting.m_Lights[i].m_SkyRadius;
		gVec3ToFloat3(&pScene->_lighting.m_Lights[i].m_P, &cl.m_Lights[i].m_P);
		gVec3ToFloat3(&pScene->_lighting.m_Lights[i].m_Target, &cl.m_Lights[i].m_Target);
		gVec3ToFloat3(&pScene->_lighting.m_Lights[i].m_N, &cl.m_Lights[i].m_N);
		gVec3ToFloat3(&pScene->_lighting.m_Lights[i].m_U, &cl.m_Lights[i].m_U);
		gVec3ToFloat3(&pScene->_lighting.m_Lights[i].m_V, &cl.m_Lights[i].m_V);
		cl.m_Lights[i].m_Area = pScene->_lighting.m_Lights[i].m_Area;
		cl.m_Lights[i].m_AreaPdf = pScene->_lighting.m_Lights[i].m_AreaPdf;
		gVec3ToFloat3(&pScene->_lighting.m_Lights[i].m_Color, &cl.m_Lights[i].m_Color);
		gVec3ToFloat3(&pScene->_lighting.m_Lights[i].m_ColorTop, &cl.m_Lights[i].m_ColorTop);
		gVec3ToFloat3(&pScene->_lighting.m_Lights[i].m_ColorMiddle, &cl.m_Lights[i].m_ColorMiddle);
		gVec3ToFloat3(&pScene->_lighting.m_Lights[i].m_ColorBottom, &cl.m_Lights[i].m_ColorBottom);
		cl.m_Lights[i].m_T = pScene->_lighting.m_Lights[i].m_T;
	}
}

void RenderGLCuda::initQuad()
{
	// treat this as negligible use of gpu mem.

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

	if (_quadVertexArray == 0) {
		glGenVertexArrays(1, &_quadVertexArray);
	}
	glBindVertexArray(_quadVertexArray);
	check_gl("create and bind verts");

	if (_quadVertices == 0) {
		glGenBuffers(1, &_quadVertices);
	}
	glBindBuffer(GL_ARRAY_BUFFER, _quadVertices);
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

	if (_quadTexcoords == 0) {
		glGenBuffers(1, &_quadTexcoords);
	}
	glBindBuffer(GL_ARRAY_BUFFER, _quadTexcoords);
	glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * square_texcoords.size(), square_texcoords.data(), GL_STATIC_DRAW);
	check_gl("init texcoord data");

	std::array<GLushort, 6> square_elements
	{
		// front
		0,  1,  2,
		2,  3,  0
	};

	if (_quadIndices == 0) {
		glGenBuffers(1, &_quadIndices);
	}
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _quadIndices);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort) * square_elements.size(), square_elements.data(), GL_STATIC_DRAW);
	num_image_elements = square_elements.size();
	check_gl("init element data");

	glBindVertexArray(0);
	check_gl("unbind vtx array");
}

void RenderGLCuda::cleanUpFB()
{
	// completely destroy the cuda binding to the framebuffer texture
	if (_cudaTex) {
		HandleCudaError(cudaDestroySurfaceObject(_cudaGLSurfaceObject));
		_cudaGLSurfaceObject = 0;
		HandleCudaError(cudaGraphicsUnregisterResource(_cudaTex));
		_cudaTex = nullptr;
	}
	// destroy the framebuffer texture
	if (_fbtex) {
		glBindTexture(GL_TEXTURE_2D, 0);
		glDeleteTextures(1, &_fbtex);
		check_gl("Destroy fb texture");
		_fbtex = 0;
	}
	if (_randomSeeds1) {
		HandleCudaError(cudaFree(_randomSeeds1));
		_randomSeeds1 = nullptr;
	}
	if (_randomSeeds2) {
		HandleCudaError(cudaFree(_randomSeeds2));
		_randomSeeds2 = nullptr;
	}
	if (_cudaF32Buffer) {
		HandleCudaError(cudaFree(_cudaF32Buffer));
		_cudaF32Buffer = nullptr;
	}
	if (_cudaF32AccumBuffer) {
		HandleCudaError(cudaFree(_cudaF32AccumBuffer));
		_cudaF32AccumBuffer = nullptr;
	}

	_gpuBytes = 0;
}

void RenderGLCuda::initFB(uint32_t w, uint32_t h)
{
	cleanUpFB();
	
	_w = w;
	_h = h;

	HandleCudaError(cudaMalloc((void**)&_cudaF32Buffer, w*h * 4 * sizeof(float)));
	HandleCudaError(cudaMemset(_cudaF32Buffer, 0, w*h * 4 * sizeof(float)));
	_gpuBytes += w*h * 4 * sizeof(float);
	HandleCudaError(cudaMalloc((void**)&_cudaF32AccumBuffer, w*h * 4 * sizeof(float)));
	HandleCudaError(cudaMemset(_cudaF32AccumBuffer, 0, w*h * 4 * sizeof(float)));
	_gpuBytes += w*h * 4 * sizeof(float);

	{
		unsigned int* pSeeds = (unsigned int*)malloc(w*h * sizeof(unsigned int));

		HandleCudaError(cudaMalloc((void**)&_randomSeeds1, w*h * sizeof(unsigned int)));
		memset(pSeeds, 0, w*h * sizeof(unsigned int));
		for (unsigned int i = 0; i < w*h; i++)
			pSeeds[i] = rand();
		HandleCudaError(cudaMemcpy(_randomSeeds1, pSeeds, w*h * sizeof(unsigned int), cudaMemcpyHostToDevice));
		_gpuBytes += w*h * sizeof(unsigned int);

		HandleCudaError(cudaMalloc((void**)&_randomSeeds2, w*h * sizeof(unsigned int)));
		memset(pSeeds, 0, w*h * sizeof(unsigned int));
		for (unsigned int i = 0; i < w*h; i++)
			pSeeds[i] = rand();
		HandleCudaError(cudaMemcpy(_randomSeeds2, pSeeds, w*h * sizeof(unsigned int), cudaMemcpyHostToDevice));
		_gpuBytes += w*h * sizeof(unsigned int);

		free(pSeeds);
	}

	glGenTextures(1, &_fbtex);
	check_gl("Gen fb texture id");
	glBindTexture(GL_TEXTURE_2D, _fbtex);
	check_gl("Bind fb texture");
	//glTextureStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, w, h);


	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	_gpuBytes += w*h * 4;
	check_gl("Create fb texture");
	// this is required in order to "complete" the texture object for mipmapless shader access.
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	// unbind the texture before doing cuda stuff.
	glBindTexture(GL_TEXTURE_2D, 0);
	
	// use gl interop to let cuda write to this tex.
	HandleCudaError(cudaGraphicsGLRegisterImage(&_cudaTex, _fbtex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

	HandleCudaError(cudaGraphicsMapResources(1, &_cudaTex));
	cudaArray_t ca;
	HandleCudaError(cudaGraphicsSubResourceGetMappedArray(&ca, _cudaTex, 0, 0));
	cudaResourceDesc desc;
	memset(&desc, 0, sizeof(desc));
	desc.resType = cudaResourceTypeArray;
	desc.res.array.array = ca;
	HandleCudaError(cudaCreateSurfaceObject(&_cudaGLSurfaceObject, &desc));
	HandleCudaError(cudaGraphicsUnmapResources(1, &_cudaTex));
}

void RenderGLCuda::initVolumeTextureCUDA() {
	// free the gpu resources of the old image.
	_imgCuda.deallocGpu();

	if (!_scene || !_scene->_volume) {
		return;
	}
	ImageCuda cimg;
	cimg.allocGpuInterleaved(_scene->_volume.get());
	_imgCuda = cimg;

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

void RenderGLCuda::doRender(const CCamera& camera) {
	if (!_scene || !_scene->_volume) {
		return;
	}
	if (!_imgCuda._volumeArrayInterleaved || _renderSettings->m_DirtyFlags.HasFlag(VolumeDirty)) {
		initVolumeTextureCUDA();
		// we have set up everything there is to do before rendering
		_status.SetRenderBegin();
	}

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
	if (_renderSettings->m_DirtyFlags.HasFlag(CameraDirty | LightsDirty | RenderParamsDirty | TransferFunctionDirty | RoiDirty))
	{
		if (_renderSettings->m_DirtyFlags.HasFlag(TransferFunctionDirty)) {
			// TODO: only update the ones that changed.
			int NC = _scene->_volume->sizeC();
			for (int i = 0; i < NC; ++i) {
				_imgCuda.updateLutGpu(i, _scene->_volume.get());
			}
		}

		//		ResetRenderCanvasView();

		// Reset no. iterations
		_renderSettings->SetNoIterations(0);
	}
	if (_renderSettings->m_DirtyFlags.HasFlag(LightsDirty)) {
		for (int i = 0; i < _scene->_lighting.m_NoLights; ++i) {
			_scene->_lighting.m_Lights[i].Update(_scene->_boundingBox);
		}

		// Reset no. iterations
		_renderSettings->SetNoIterations(0);
	}
	if (_renderSettings->m_DirtyFlags.HasFlag(VolumeDataDirty))
	{
		int ch[4] = { 0, 0, 0, 0 };
		int activeChannel = 0;
		int NC = _scene->_volume->sizeC();
		for (int i = 0; i < NC; ++i) {
			if (_scene->_material.enabled[i] && activeChannel < 4) {
				ch[activeChannel] = i;
				activeChannel++;
			}
		}
		_imgCuda.updateVolumeData4x16(_scene->_volume.get(), ch[0], ch[1], ch[2], ch[3]);
		_renderSettings->SetNoIterations(0);
	}
	// At this point, all dirty flags should have been taken care of, since the flags in the original scene are now cleared
	_renderSettings->m_DirtyFlags.ClearAllFlags();

	_renderSettings->m_RenderSettings.m_GradientDelta = 1.0f / (float)this->_scene->_volume->maxPixelDimension();

	_renderSettings->m_DenoiseParams.SetWindowRadius(3.0f);

	CudaLighting cudalt;
	FillCudaLighting(_scene, cudalt);
    CudaCamera cudacam;
    FillCudaCamera(&(camera), cudacam);

	glm::vec3 sn = _scene->_boundingBox.GetMinP();
	glm::vec3 ext = _scene->_boundingBox.GetExtent();
	CBoundingBox b;
	b.SetMinP(glm::vec3(
		ext.x*_scene->_roi.GetMinP().x + sn.x,
		ext.y*_scene->_roi.GetMinP().y + sn.y,
		ext.z*_scene->_roi.GetMinP().z + sn.z
	));
	b.SetMaxP(glm::vec3(
		ext.x*_scene->_roi.GetMaxP().x + sn.x,
		ext.y*_scene->_roi.GetMaxP().y + sn.y,
		ext.z*_scene->_roi.GetMaxP().z + sn.z
	));


	BindConstants(cudalt, _renderSettings->m_DenoiseParams, cudacam,
		_scene->_boundingBox, b, _renderSettings->m_RenderSettings, _renderSettings->GetNoIterations(),
		_w, _h, camera.m_Film.m_Gamma, camera.m_Film.m_Exposure);
	// Render image
	//RayMarchVolume(_cudaF32Buffer, _volumeTex, _volumeGradientTex, _renderSettings, _w, _h, 2.0f, 20.0f, glm::value_ptr(m), _channelMin, _channelMax);
	cudaFB theCudaFB = {
		_cudaF32Buffer,
		_cudaF32AccumBuffer,
		_randomSeeds1,
		_randomSeeds2
	};

	// single channel
	int NC = _scene->_volume->sizeC();
	// use first 3 channels only.
	int activeChannel = 0;
	cudaVolume theCudaVolume(0);
	for (int i = 0; i < NC; ++i) {
		if (_scene->_material.enabled[i] && activeChannel < MAX_CUDA_CHANNELS) {
			theCudaVolume.volumeTexture[activeChannel] = _imgCuda._volumeTextureInterleaved;
			theCudaVolume.gradientVolumeTexture[activeChannel] = _imgCuda._channels[i]._volumeGradientTexture;
			theCudaVolume.lutTexture[activeChannel] = _imgCuda._channels[i]._volumeLutTexture;
			theCudaVolume.intensityMax[activeChannel] = _scene->_volume->channel(i)->_max;
			theCudaVolume.diffuse[activeChannel * 3 + 0] = _scene->_material.diffuse[i * 3 + 0];
			theCudaVolume.diffuse[activeChannel * 3 + 1] = _scene->_material.diffuse[i * 3 + 1];
			theCudaVolume.diffuse[activeChannel * 3 + 2] = _scene->_material.diffuse[i * 3 + 2];
			theCudaVolume.specular[activeChannel * 3 + 0] = _scene->_material.specular[i * 3 + 0];
			theCudaVolume.specular[activeChannel * 3 + 1] = _scene->_material.specular[i * 3 + 1];
			theCudaVolume.specular[activeChannel * 3 + 2] = _scene->_material.specular[i * 3 + 2];
			theCudaVolume.emissive[activeChannel * 3 + 0] = _scene->_material.emissive[i * 3 + 0];
			theCudaVolume.emissive[activeChannel * 3 + 1] = _scene->_material.emissive[i * 3 + 1];
			theCudaVolume.emissive[activeChannel * 3 + 2] = _scene->_material.emissive[i * 3 + 2];
			theCudaVolume.roughness[activeChannel] = _scene->_material.roughness[i];

			activeChannel++;
			theCudaVolume.nChannels = activeChannel;
		}
	}

	int numIterations = _renderSettings->GetNoIterations();
	Render(0, camera.m_Film.m_ExposureIterations, 
        camera.m_Film.m_Resolution.GetResX(), 
        camera.m_Film.m_Resolution.GetResY(),
		theCudaFB,
		theCudaVolume,
		_timingRender, _timingBlur, _timingPostProcess, _timingDenoise, numIterations);
	_renderSettings->SetNoIterations(numIterations);
	//LOG_DEBUG << "RETURN FROM RENDER";

	// Tonemap into opengl display buffer

	// do cuda with cudaSurfaceObj

	// set the lerpC here because the Render call is incrementing the number of iterations.
	//_renderSettings->m_DenoiseParams.m_LerpC = 0.33f * (max((float)_renderSettings->GetNoIterations(), 1.0f) * 1.0f);//1.0f - powf(1.0f / (float)gScene.GetNoIterations(), 15.0f);//1.0f - expf(-0.01f * (float)gScene.GetNoIterations());
	_renderSettings->m_DenoiseParams.m_LerpC = 0.33f * (max((float)_renderSettings->GetNoIterations(), 1.0f) * 0.035f);//1.0f - powf(1.0f / (float)gScene.GetNoIterations(), 15.0f);//1.0f - expf(-0.01f * (float)gScene.GetNoIterations());
//	LOG_DEBUG << "Window " << _w << " " << _h << " Cam " << _renderSettings->m_Camera.m_Film.m_Resolution.GetResX() << " " << _renderSettings->m_Camera.m_Film.m_Resolution.GetResY();
	CCudaTimer TmrDenoise;
	if (_renderSettings->m_DenoiseParams.m_Enabled && _renderSettings->m_DenoiseParams.m_LerpC > 0.0f && _renderSettings->m_DenoiseParams.m_LerpC < 1.0f)
	{
		Denoise(_cudaF32AccumBuffer, _cudaGLSurfaceObject, _w, _h, _renderSettings->m_DenoiseParams.m_LerpC);
	}
	else
	{
		ToneMap(_cudaF32AccumBuffer, _cudaGLSurfaceObject, _w, _h);
	}
	_timingDenoise.AddDuration(TmrDenoise.ElapsedTime());
	
	HandleCudaError(cudaStreamSynchronize(0));
	
	// display timings.
	
	_status.SetStatisticChanged("Performance", "Render Image", QString::number(_timingRender.m_FilteredDuration, 'f', 2), "ms.");
	_status.SetStatisticChanged("Performance", "Blur Estimate", QString::number(_timingBlur.m_FilteredDuration, 'f', 2), "ms.");
	_status.SetStatisticChanged("Performance", "Post Process Estimate", QString::number(_timingPostProcess.m_FilteredDuration, 'f', 2), "ms.");
	_status.SetStatisticChanged("Performance", "De-noise Image", QString::number(_timingDenoise.m_FilteredDuration, 'f', 2), "ms.");

	//FPS.AddDuration(1000.0f / TmrFps.ElapsedTime());

	//_status.SetStatisticChanged("Performance", "FPS", QString::number(FPS.m_FilteredDuration, 'f', 2), "Frames/Sec.");
	_status.SetStatisticChanged("Performance", "No. Iterations", QString::number(_renderSettings->GetNoIterations()), "Iterations");
	
}

void RenderGLCuda::render(const CCamera& camera)
{
	// draw to _fbtex
	doRender(camera);

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

	glBindVertexArray(_quadVertexArray);
	check_gl("bind vtx buf");

	image_shader->enableCoords();
	image_shader->setCoords(_quadVertices, 0, 2);

	image_shader->enableTexCoords();
	image_shader->setTexCoords(_quadTexcoords, 0, 2);

	// Push each element to the vertex shader
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _quadIndices);
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
	//w = 8; h = 8;
	glViewport(0, 0, w, h);
	if ((_w == w) && (_h == h)) {
		return;
	}

	initFB(w, h);
	LOG_DEBUG << "Resized window to " << w << " x " << h;
}

void RenderGLCuda::cleanUpQuad() {
	glDeleteVertexArrays(1, &_quadVertexArray);
	_quadVertexArray = 0;
	glDeleteBuffers(1, &_quadVertices);
	_quadVertices = 0;
	glDeleteBuffers(1, &_quadTexcoords);
	_quadTexcoords = 0;
	glDeleteBuffers(1, &_quadIndices);
	_quadIndices = 0;
}

void RenderGLCuda::cleanUpResources() {
	_imgCuda.deallocGpu();

	cleanUpQuad();
	cleanUpFB();

}

RenderParams& RenderGLCuda::renderParams() {
	return _renderParams;
}
Scene* RenderGLCuda::scene() {
	return _scene;
}
void RenderGLCuda::setScene(Scene* s) {
	_scene = s;
}

size_t RenderGLCuda::getGpuBytes() {
	return _gpuBytes + _imgCuda._gpuBytes;
}
