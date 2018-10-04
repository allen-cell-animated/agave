#include "RenderGLPT.h"

#include "glad/glad.h"
#include "glm.h"

#include "gl/Util.h"
#include "gl/v33/V33Image3D.h"
#include "gl/v33/V33FSQ.h"
#include "glsl/v330/V330GLImageShader2DnoLut.h"
#include "glsl/v330/GLPTAccumShader.h"
#include "glsl/v330/GLPTVolumeShader.h"
#include "ImageXYZC.h"
#include "Logging.h"
#include "cudarndr/RenderThread.h"

#include "Core.cuh"
#include "Lighting2.cuh"
#include "Camera2.cuh"

#include <array>

RenderGLPT::RenderGLPT(RenderSettings* rs)
	:_glF32Buffer(0),
    _glF32AccumBuffer(0),
    _glF32AccumBuffer2(0),
    _fbF32(0),
    _fbF32Accum(0),
	_fbtex(0),
    _renderBufferShader(nullptr),
    _accumBufferShader(nullptr),
    _fsq(nullptr),
	_randomSeeds1(nullptr),
	_randomSeeds2(nullptr),
	_renderSettings(rs),
	_w(0),
	_h(0),
	_scene(nullptr),
	_gpuBytes(0),
	_imagequad(nullptr)
{
}


RenderGLPT::~RenderGLPT()
{
}

void RenderGLPT::FillCudaCamera(const CCamera* pCamera, CudaCamera& c) {
//    gVec3ToFloat3(&pCamera->m_From, &c.m_From);
//    gVec3ToFloat3(&pCamera->m_N, &c.m_N);
//    gVec3ToFloat3(&pCamera->m_U, &c.m_U);
//    gVec3ToFloat3(&pCamera->m_V, &c.m_V);
    c.m_ApertureSize = pCamera->m_Aperture.m_Size;
    c.m_FocalDistance = pCamera->m_Focus.m_FocalDistance;
    c.m_InvScreen[0] = pCamera->m_Film.m_InvScreen.x;
    c.m_InvScreen[1] = pCamera->m_Film.m_InvScreen.y;
    c.m_Screen[0][0] = pCamera->m_Film.m_Screen[0][0];
    c.m_Screen[1][0] = pCamera->m_Film.m_Screen[1][0];
    c.m_Screen[0][1] = pCamera->m_Film.m_Screen[0][1];
    c.m_Screen[1][1] = pCamera->m_Film.m_Screen[1][1];
}

void RenderGLPT::FillCudaLighting(Scene* pScene, CudaLighting& cl) {
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
//		gVec3ToFloat3(&pScene->_lighting.m_Lights[i].m_P, &cl.m_Lights[i].m_P);
//		gVec3ToFloat3(&pScene->_lighting.m_Lights[i].m_Target, &cl.m_Lights[i].m_Target);
//		gVec3ToFloat3(&pScene->_lighting.m_Lights[i].m_N, &cl.m_Lights[i].m_N);
//		gVec3ToFloat3(&pScene->_lighting.m_Lights[i].m_U, &cl.m_Lights[i].m_U);
//		gVec3ToFloat3(&pScene->_lighting.m_Lights[i].m_V, &cl.m_Lights[i].m_V);
		cl.m_Lights[i].m_Area = pScene->_lighting.m_Lights[i].m_Area;
		cl.m_Lights[i].m_AreaPdf = pScene->_lighting.m_Lights[i].m_AreaPdf;
//		gVec3ToFloat3(&pScene->_lighting.m_Lights[i].m_Color, &cl.m_Lights[i].m_Color);
//		gVec3ToFloat3(&pScene->_lighting.m_Lights[i].m_ColorTop, &cl.m_Lights[i].m_ColorTop);
//		gVec3ToFloat3(&pScene->_lighting.m_Lights[i].m_ColorMiddle, &cl.m_Lights[i].m_ColorMiddle);
//		gVec3ToFloat3(&pScene->_lighting.m_Lights[i].m_ColorBottom, &cl.m_Lights[i].m_ColorBottom);
		cl.m_Lights[i].m_T = pScene->_lighting.m_Lights[i].m_T;
	}
}

void RenderGLPT::cleanUpFB()
{
	// destroy the framebuffer texture
	if (_fbtex) {
		glBindTexture(GL_TEXTURE_2D, 0);
		glDeleteTextures(1, &_fbtex);
		check_gl("Destroy fb texture");
		_fbtex = 0;
	}
    if (_fb) {
        glDeleteFramebuffers(1, &_fb);
        _fb = 0;
    }
    if (_randomSeeds1) {
		HandleCudaError(cudaFree(_randomSeeds1));
		_randomSeeds1 = nullptr;
	}
	if (_randomSeeds2) {
		HandleCudaError(cudaFree(_randomSeeds2));
		_randomSeeds2 = nullptr;
	}

    if (_fbF32) {
        glDeleteFramebuffers(1, &_fbF32);
        _fbF32 = 0;
    }
    if (_glF32Buffer) {
        glBindTexture(GL_TEXTURE_2D, 0);
        glDeleteTextures(1, &_glF32Buffer);
        check_gl("Destroy fb texture");
        _glF32Buffer = 0;
    }
    if (_fbF32Accum) {
        glDeleteFramebuffers(1, &_fbF32Accum);
        _fbF32Accum = 0;
    }
    if (_glF32AccumBuffer) {
        glBindTexture(GL_TEXTURE_2D, 0);
        glDeleteTextures(1, &_glF32AccumBuffer);
        check_gl("Destroy fb texture");
        _glF32AccumBuffer = 0;
    }
    if (_glF32AccumBuffer2) {
        glBindTexture(GL_TEXTURE_2D, 0);
        glDeleteTextures(1, &_glF32AccumBuffer2);
        check_gl("Destroy fb2 texture");
        _glF32AccumBuffer2 = 0;
    }

    delete _renderBufferShader;
    _renderBufferShader = 0;
    delete _accumBufferShader;
    _accumBufferShader = 0;
    delete _fsq;
    _fsq = 0;

	_gpuBytes = 0;
}

void RenderGLPT::initFB(uint32_t w, uint32_t h)
{
	cleanUpFB();
	
    glGenTextures(1, &_glF32Buffer);
    check_gl("Gen fb texture id");
    glBindTexture(GL_TEXTURE_2D, _glF32Buffer);
    check_gl("Bind fb texture");
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    _gpuBytes += w*h * 4* sizeof(float);
    check_gl("Create fb texture");

    glGenFramebuffers(1, &_fbF32);
    glBindFramebuffer(GL_FRAMEBUFFER, _fbF32);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _glF32Buffer, 0);


    glGenTextures(1, &_glF32AccumBuffer);
    check_gl("Gen fb texture id");
    glBindTexture(GL_TEXTURE_2D, _glF32AccumBuffer);
    check_gl("Bind fb texture");
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    _gpuBytes += w*h * 4 * sizeof(float);
    check_gl("Create fb texture");

    glGenTextures(1, &_glF32AccumBuffer2);
    check_gl("Gen fb2 texture id");
    glBindTexture(GL_TEXTURE_2D, _glF32AccumBuffer2);
    check_gl("Bind fb2 texture");
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    _gpuBytes += w*h * 4 * sizeof(float);
    check_gl("Create fb2 texture");

    glGenFramebuffers(1, &_fbF32Accum);
    glBindFramebuffer(GL_FRAMEBUFFER, _fbF32Accum);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _glF32AccumBuffer, 0);

    _fsq = new FSQ();
    _fsq->setSize(glm::vec2(-1, 1), glm::vec2(-1, 1));
    _fsq->create();
    _renderBufferShader = new GLPTVolumeShader();
    _accumBufferShader = new GLPTAccumShader();
    
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

    glGenFramebuffers(1, &_fb);
    glBindFramebuffer(GL_FRAMEBUFFER, _fb);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, _fbtex, 0);

}

void RenderGLPT::initVolumeTextureCUDA() {
    // free the gpu resources of the old image.
    _imgCuda.deallocGpu();

    if (!_scene || !_scene->_volume) {
        return;
    }
    ImageGL cimg;
    cimg.allocGpuInterleaved(_scene->_volume.get());
    _imgCuda = cimg;

}

void RenderGLPT::initialize(uint32_t w, uint32_t h)
{
	_imagequad = new RectImage2D();

	initVolumeTextureCUDA();
    check_gl("init gl volume");

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	check_gl("init gl state");

	// Size viewport
	resize(w,h);
}

void RenderGLPT::doRender(const CCamera& camera) {
	if (!_scene || !_scene->_volume) {
		return;
	}

    if (!_imgCuda._volumeTextureInterleaved || _renderSettings->m_DirtyFlags.HasFlag(VolumeDirty)) {
		initVolumeTextureCUDA();
		// we have set up everything there is to do before rendering
		_status.SetRenderBegin();
	}

	// Resizing the image canvas requires special attention
	if (_renderSettings->m_DirtyFlags.HasFlag(FilmResolutionDirty))
	{
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


	//BindConstants(cudalt, _renderSettings->m_DenoiseParams, cudacam,
	//	_scene->_boundingBox, b, _renderSettings->m_RenderSettings, _renderSettings->GetNoIterations(),
	//	_w, _h, camera.m_Film.m_Gamma, camera.m_Film.m_Exposure);

    
    // Render image
	//RayMarchVolume(_cudaF32Buffer, _volumeTex, _volumeGradientTex, _renderSettings, _w, _h, 2.0f, 20.0f, glm::value_ptr(m), _channelMin, _channelMax);
//	cudaFB theCudaFB = {
//		_cudaF32Buffer,
//		_cudaF32AccumBuffer,
//		_randomSeeds1,
//		_randomSeeds2
//	};


	// find nearest intersection to set camera focal distance automatically.
	// then re-upload that data.
//	if (camera.m_Focus.m_Type == 0) {
//		ComputeFocusDistance(theCudaVolume);
//	}

	int numIterations = _renderSettings->GetNoIterations();

    glm::mat4 m(1.0);
    // set all the vars


    GLuint accumTargetTex = numIterations % 2 ? _glF32AccumBuffer : _glF32AccumBuffer2;
    GLuint prevAccumTargetTex = numIterations % 2 ? _glF32AccumBuffer2 : _glF32AccumBuffer;

    for (int i = 0; i < camera.m_Film.m_ExposureIterations; ++i) {
        //CCudaTimer TmrRender;

        // set the pt shader
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_3D, _imgCuda._volumeTextureInterleaved);

        // draw fullscreen quad
        glBindFramebuffer(GL_FRAMEBUFFER, _fbF32);
        _renderBufferShader->bind();
        _renderBufferShader->volumeTexture = _imgCuda._volumeTextureInterleaved;
        // single channel
        int NC = _scene->_volume->sizeC();
        // use first 3 channels only.
        int activeChannel = 0;
        for (int i = 0; i < NC; ++i) {
            if (_scene->_material.enabled[i] && activeChannel < MAX_GL_CHANNELS) {
                _renderBufferShader->lutTexture[activeChannel] = _imgCuda._channels[i]._volumeLutTexture;
                _renderBufferShader->intensityMax[activeChannel] = _scene->_volume->channel(i)->_max;
                _renderBufferShader->intensityMin[activeChannel] = _scene->_volume->channel(i)->_min;
                _renderBufferShader->diffuse[activeChannel * 3 + 0] = _scene->_material.diffuse[i * 3 + 0];
                _renderBufferShader->diffuse[activeChannel * 3 + 1] = _scene->_material.diffuse[i * 3 + 1];
                _renderBufferShader->diffuse[activeChannel * 3 + 2] = _scene->_material.diffuse[i * 3 + 2];
                _renderBufferShader->specular[activeChannel * 3 + 0] = _scene->_material.specular[i * 3 + 0];
                _renderBufferShader->specular[activeChannel * 3 + 1] = _scene->_material.specular[i * 3 + 1];
                _renderBufferShader->specular[activeChannel * 3 + 2] = _scene->_material.specular[i * 3 + 2];
                _renderBufferShader->emissive[activeChannel * 3 + 0] = _scene->_material.emissive[i * 3 + 0];
                _renderBufferShader->emissive[activeChannel * 3 + 1] = _scene->_material.emissive[i * 3 + 1];
                _renderBufferShader->emissive[activeChannel * 3 + 2] = _scene->_material.emissive[i * 3 + 2];
                _renderBufferShader->roughness[activeChannel] = _scene->_material.roughness[i];
                _renderBufferShader->opacity[activeChannel] = _scene->_material.opacity[i];

                activeChannel++;
                _renderBufferShader->nChannels = activeChannel;
            }
        }



        _renderBufferShader->setShadingUniforms();

        _fsq->render(m);
        //_timingRender.AddDuration(TmrRender.ElapsedTime());

        // estimate just adds to accumulation buffer.
        //CCudaTimer TmrPostProcess;
        // accumulate
        glBindFramebuffer(GL_FRAMEBUFFER, _fbF32Accum);
        
        accumTargetTex = numIterations % 2 ? _glF32AccumBuffer : _glF32AccumBuffer2;
        prevAccumTargetTex = numIterations % 2 ? _glF32AccumBuffer2 : _glF32AccumBuffer;
        
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, accumTargetTex, 0);

        // the sample
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, _glF32Buffer);
        // the accum buffer
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, prevAccumTargetTex);

        _accumBufferShader->bind();
        _accumBufferShader->numIterations = numIterations;
        _accumBufferShader->setShadingUniforms();
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        _fsq->render(m);
        glEnable(GL_BLEND);
        glEnable(GL_DEPTH_TEST);
        _accumBufferShader->release();
        //_timingPostProcess.AddDuration(TmrPostProcess.ElapsedTime());

        // ping pong accum buffer. this will stall till previous accum render is done.

        numIterations++;
        const float NoIterations = numIterations;
        const float InvNoIterations = 1.0f / ((NoIterations > 1.0f) ? NoIterations : 1.0f);
        //HandleCudaError(cudaMemcpyToSymbol(gNoIterations, &NoIterations, sizeof(float)));
        //HandleCudaError(cudaMemcpyToSymbol(gInvNoIterations, &InvNoIterations, sizeof(float)));
    }

    glBindFramebuffer(GL_FRAMEBUFFER, _fb);

    // tone map into rgba8 buffer.
    _imagequad->draw(prevAccumTargetTex);



//    Render(0, camera.m_Film.m_ExposureIterations,
//        camera.m_Film.m_Resolution.GetResX(), 
//        camera.m_Film.m_Resolution.GetResY(),
//		theCudaFB,
//		theCudaVolume,
//		_timingRender, _timingBlur, _timingPostProcess, _timingDenoise, numIterations);

	_renderSettings->SetNoIterations(numIterations);
	//LOG_DEBUG << "RETURN FROM RENDER";

	// Tonemap into opengl display buffer


	// set the lerpC here because the Render call is incrementing the number of iterations.
	//_renderSettings->m_DenoiseParams.m_LerpC = 0.33f * (max((float)_renderSettings->GetNoIterations(), 1.0f) * 1.0f);//1.0f - powf(1.0f / (float)gScene.GetNoIterations(), 15.0f);//1.0f - expf(-0.01f * (float)gScene.GetNoIterations());
	_renderSettings->m_DenoiseParams.m_LerpC = 0.33f * (max((float)_renderSettings->GetNoIterations(), 1.0f) * 0.035f);//1.0f - powf(1.0f / (float)gScene.GetNoIterations(), 15.0f);//1.0f - expf(-0.01f * (float)gScene.GetNoIterations());
//	LOG_DEBUG << "Window " << _w << " " << _h << " Cam " << _renderSettings->m_Camera.m_Film.m_Resolution.GetResX() << " " << _renderSettings->m_Camera.m_Film.m_Resolution.GetResY();
	//CCudaTimer TmrDenoise;
	if (_renderSettings->m_DenoiseParams.m_Enabled && _renderSettings->m_DenoiseParams.m_LerpC > 0.0f && _renderSettings->m_DenoiseParams.m_LerpC < 1.0f)
	{
        // draw from accum buffer into fbtex
		// Denoise(_cudaF32AccumBuffer, _fbTex, _w, _h, _renderSettings->m_DenoiseParams.m_LerpC);
	}
	else
	{
		// ToneMap(_cudaF32AccumBuffer, _fbTex, _w, _h);
	}
	//_timingDenoise.AddDuration(TmrDenoise.ElapsedTime());
		
	// display timings.
	
	_status.SetStatisticChanged("Performance", "Render Image", QString::number(_timingRender.m_FilteredDuration, 'f', 2), "ms.");
	_status.SetStatisticChanged("Performance", "Blur Estimate", QString::number(_timingBlur.m_FilteredDuration, 'f', 2), "ms.");
	_status.SetStatisticChanged("Performance", "Post Process Estimate", QString::number(_timingPostProcess.m_FilteredDuration, 'f', 2), "ms.");
	_status.SetStatisticChanged("Performance", "De-noise Image", QString::number(_timingDenoise.m_FilteredDuration, 'f', 2), "ms.");

	//FPS.AddDuration(1000.0f / TmrFps.ElapsedTime());

	//_status.SetStatisticChanged("Performance", "FPS", QString::number(FPS.m_FilteredDuration, 'f', 2), "Frames/Sec.");
	_status.SetStatisticChanged("Performance", "No. Iterations", QString::number(_renderSettings->GetNoIterations()), "Iterations");
	
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

}

void RenderGLPT::render(const CCamera& camera)
{
	// draw to _fbtex
	doRender(camera);

	// put _fbtex to main render target
	drawImage();
}

void RenderGLPT::drawImage() {
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// draw quad using the tex that cudaTex was mapped to
	_imagequad->draw(_fbtex);
}


void RenderGLPT::resize(uint32_t w, uint32_t h)
{
	//w = 8; h = 8;
	glViewport(0, 0, w, h);
	if ((_w == w) && (_h == h)) {
		return;
	}

	initFB(w, h);
	LOG_DEBUG << "Resized window to " << w << " x " << h;

	_w = w;
	_h = h;
}

void RenderGLPT::cleanUpResources() {
	_imgCuda.deallocGpu();

	delete _imagequad;
	_imagequad = nullptr;

	cleanUpFB();

}

RenderParams& RenderGLPT::renderParams() {
	return _renderParams;
}
Scene* RenderGLPT::scene() {
	return _scene;
}
void RenderGLPT::setScene(Scene* s) {
	_scene = s;
}

size_t RenderGLPT::getGpuBytes() {
	return _gpuBytes + _imgCuda._gpuBytes;
}
