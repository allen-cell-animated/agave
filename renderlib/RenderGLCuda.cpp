#include "RenderGLCuda.h"

#include "glad/glad.h"
#include "glm.h"

#include "ImageXYZC.h"
#include "Logging.h"
#include "gl/Util.h"
#include "gl/v33/V33Image3D.h"
#include "glsl/v330/V330GLImageShader2DnoLut.h"
#include "renderlib.h"

#include "Camera2.cuh"
#include "Core.cuh"
#include "Lighting2.cuh"

#include <array>

RenderGLCuda::RenderGLCuda(RenderSettings* rs)
  : m_cudaF32Buffer(nullptr)
  , m_cudaF32AccumBuffer(nullptr)
  , m_cudaTex(nullptr)
  , m_cudaGLSurfaceObject(0)
  , m_fbtex(0)
  , m_randomSeeds1(nullptr)
  , m_randomSeeds2(nullptr)
  , m_renderSettings(rs)
  , m_w(0)
  , m_h(0)
  , m_scene(nullptr)
  , m_gpuBytes(0)
  , m_imagequad(nullptr)
{}

RenderGLCuda::~RenderGLCuda() {}

void
gVec3ToFloat3(const glm::vec3* src, float3* dest)
{
  dest->x = src->x;
  dest->y = src->y;
  dest->z = src->z;
}

void
RenderGLCuda::FillCudaCamera(const CCamera* pCamera, CudaCamera& c)
{
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

void
RenderGLCuda::FillCudaLighting(Scene* pScene, CudaLighting& cl)
{
  cl.m_NoLights = pScene->m_lighting.m_NoLights;
  for (int i = 0; i < min(cl.m_NoLights, MAX_CUDA_LIGHTS); ++i) {
    cl.m_Lights[i].m_Theta = pScene->m_lighting.m_Lights[i].m_Theta;
    cl.m_Lights[i].m_Phi = pScene->m_lighting.m_Lights[i].m_Phi;
    cl.m_Lights[i].m_Width = pScene->m_lighting.m_Lights[i].m_Width;
    cl.m_Lights[i].m_InvWidth = pScene->m_lighting.m_Lights[i].m_InvWidth;
    cl.m_Lights[i].m_HalfWidth = pScene->m_lighting.m_Lights[i].m_HalfWidth;
    cl.m_Lights[i].m_InvHalfWidth = pScene->m_lighting.m_Lights[i].m_InvHalfWidth;
    cl.m_Lights[i].m_Height = pScene->m_lighting.m_Lights[i].m_Height;
    cl.m_Lights[i].m_InvHeight = pScene->m_lighting.m_Lights[i].m_InvHeight;
    cl.m_Lights[i].m_HalfHeight = pScene->m_lighting.m_Lights[i].m_HalfHeight;
    cl.m_Lights[i].m_InvHalfHeight = pScene->m_lighting.m_Lights[i].m_InvHalfHeight;
    cl.m_Lights[i].m_Distance = pScene->m_lighting.m_Lights[i].m_Distance;
    cl.m_Lights[i].m_SkyRadius = pScene->m_lighting.m_Lights[i].m_SkyRadius;
    gVec3ToFloat3(&pScene->m_lighting.m_Lights[i].m_P, &cl.m_Lights[i].m_P);
    gVec3ToFloat3(&pScene->m_lighting.m_Lights[i].m_Target, &cl.m_Lights[i].m_Target);
    gVec3ToFloat3(&pScene->m_lighting.m_Lights[i].m_N, &cl.m_Lights[i].m_N);
    gVec3ToFloat3(&pScene->m_lighting.m_Lights[i].m_U, &cl.m_Lights[i].m_U);
    gVec3ToFloat3(&pScene->m_lighting.m_Lights[i].m_V, &cl.m_Lights[i].m_V);
    cl.m_Lights[i].m_Area = pScene->m_lighting.m_Lights[i].m_Area;
    cl.m_Lights[i].m_AreaPdf = pScene->m_lighting.m_Lights[i].m_AreaPdf;
    gVec3ToFloat3(&pScene->m_lighting.m_Lights[i].m_Color, &cl.m_Lights[i].m_Color);
    cl.m_Lights[i].m_Color *= pScene->m_lighting.m_Lights[i].m_ColorIntensity;
    gVec3ToFloat3(&pScene->m_lighting.m_Lights[i].m_ColorTop, &cl.m_Lights[i].m_ColorTop);
    cl.m_Lights[i].m_ColorTop *= pScene->m_lighting.m_Lights[i].m_ColorTopIntensity;
    gVec3ToFloat3(&pScene->m_lighting.m_Lights[i].m_ColorMiddle, &cl.m_Lights[i].m_ColorMiddle);
    cl.m_Lights[i].m_ColorMiddle *= pScene->m_lighting.m_Lights[i].m_ColorMiddleIntensity;
    gVec3ToFloat3(&pScene->m_lighting.m_Lights[i].m_ColorBottom, &cl.m_Lights[i].m_ColorBottom);
    cl.m_Lights[i].m_ColorBottom *= pScene->m_lighting.m_Lights[i].m_ColorBottomIntensity;
    cl.m_Lights[i].m_T = pScene->m_lighting.m_Lights[i].m_T;
  }
}

void
RenderGLCuda::cleanUpFB()
{
  // completely destroy the cuda binding to the framebuffer texture
  if (m_cudaTex) {
    HandleCudaError(cudaDestroySurfaceObject(m_cudaGLSurfaceObject));
    m_cudaGLSurfaceObject = 0;
    HandleCudaError(cudaGraphicsUnregisterResource(m_cudaTex));
    m_cudaTex = nullptr;
  }
  // destroy the framebuffer texture
  if (m_fbtex) {
    glBindTexture(GL_TEXTURE_2D, 0);
    glDeleteTextures(1, &m_fbtex);
    check_gl("Destroy fb texture");
    m_fbtex = 0;
  }
  if (m_randomSeeds1) {
    HandleCudaError(cudaFree(m_randomSeeds1));
    m_randomSeeds1 = nullptr;
  }
  if (m_randomSeeds2) {
    HandleCudaError(cudaFree(m_randomSeeds2));
    m_randomSeeds2 = nullptr;
  }
  if (m_cudaF32Buffer) {
    HandleCudaError(cudaFree(m_cudaF32Buffer));
    m_cudaF32Buffer = nullptr;
  }
  if (m_cudaF32AccumBuffer) {
    HandleCudaError(cudaFree(m_cudaF32AccumBuffer));
    m_cudaF32AccumBuffer = nullptr;
  }

  m_gpuBytes = 0;
}

void
RenderGLCuda::initFB(uint32_t w, uint32_t h)
{
  cleanUpFB();

  HandleCudaError(cudaMalloc((void**)&m_cudaF32Buffer, w * h * 4 * sizeof(float)));
  HandleCudaError(cudaMemset(m_cudaF32Buffer, 0, w * h * 4 * sizeof(float)));
  m_gpuBytes += w * h * 4 * sizeof(float);
  HandleCudaError(cudaMalloc((void**)&m_cudaF32AccumBuffer, w * h * 4 * sizeof(float)));
  HandleCudaError(cudaMemset(m_cudaF32AccumBuffer, 0, w * h * 4 * sizeof(float)));
  m_gpuBytes += w * h * 4 * sizeof(float);

  {
    unsigned int* pSeeds = (unsigned int*)malloc(w * h * sizeof(unsigned int));

    HandleCudaError(cudaMalloc((void**)&m_randomSeeds1, w * h * sizeof(unsigned int)));
    memset(pSeeds, 0, w * h * sizeof(unsigned int));
    for (unsigned int i = 0; i < w * h; i++)
      pSeeds[i] = rand();
    HandleCudaError(cudaMemcpy(m_randomSeeds1, pSeeds, w * h * sizeof(unsigned int), cudaMemcpyHostToDevice));
    m_gpuBytes += w * h * sizeof(unsigned int);

    HandleCudaError(cudaMalloc((void**)&m_randomSeeds2, w * h * sizeof(unsigned int)));
    memset(pSeeds, 0, w * h * sizeof(unsigned int));
    for (unsigned int i = 0; i < w * h; i++)
      pSeeds[i] = rand();
    HandleCudaError(cudaMemcpy(m_randomSeeds2, pSeeds, w * h * sizeof(unsigned int), cudaMemcpyHostToDevice));
    m_gpuBytes += w * h * sizeof(unsigned int);

    free(pSeeds);
  }

  glGenTextures(1, &m_fbtex);
  check_gl("Gen fb texture id");
  glBindTexture(GL_TEXTURE_2D, m_fbtex);
  check_gl("Bind fb texture");
  // glTextureStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, w, h);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  m_gpuBytes += w * h * 4;
  check_gl("Create fb texture");
  // this is required in order to "complete" the texture object for mipmapless shader access.
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  // unbind the texture before doing cuda stuff.
  glBindTexture(GL_TEXTURE_2D, 0);

  // use gl interop to let cuda write to this tex.
  HandleCudaError(
    cudaGraphicsGLRegisterImage(&m_cudaTex, m_fbtex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore));

  HandleCudaError(cudaGraphicsMapResources(1, &m_cudaTex));
  cudaArray_t ca;
  HandleCudaError(cudaGraphicsSubResourceGetMappedArray(&ca, m_cudaTex, 0, 0));
  cudaResourceDesc desc;
  memset(&desc, 0, sizeof(desc));
  desc.resType = cudaResourceTypeArray;
  desc.res.array.array = ca;
  HandleCudaError(cudaCreateSurfaceObject(&m_cudaGLSurfaceObject, &desc));
  HandleCudaError(cudaGraphicsUnmapResources(1, &m_cudaTex));
}

void
RenderGLCuda::initVolumeTextureCUDA()
{

  // renderlib::removeCudaImage(_imgCuda);

  // free the gpu resources of the old image.

  // if (_imgCuda) {
  //	renderlib::imageDeallocGPU_Cuda(_scene->_volume);
  //	_imgCuda.reset();
  //}

  if (m_scene && m_scene->m_volume) {
    m_imgCuda = renderlib::imageAllocGPU_Cuda(m_scene->m_volume, false);
  }
}

void
RenderGLCuda::initialize(uint32_t w, uint32_t h)
{
  m_imagequad = new RectImage2D();

  initVolumeTextureCUDA();

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  check_gl("init gl state");

  // Size viewport
  resize(w, h);
}

void
RenderGLCuda::doRender(const CCamera& camera)
{
  if (!m_scene || !m_scene->m_volume) {
    return;
  }
  if (!m_imgCuda || !m_imgCuda->m_volumeArrayInterleaved || m_renderSettings->m_DirtyFlags.HasFlag(VolumeDirty)) {
    initVolumeTextureCUDA();
    // we have set up everything there is to do before rendering
    m_status.SetRenderBegin();
  }

  // Resizing the image canvas requires special attention
  if (m_renderSettings->m_DirtyFlags.HasFlag(FilmResolutionDirty)) {
    m_renderSettings->SetNoIterations(0);

    // Log("Render canvas resized to: " + QString::number(SceneCopy.m_Camera.m_Film.m_Resolution.GetResX()) + " x " +
    // QString::number(SceneCopy.m_Camera.m_Film.m_Resolution.GetResY()) + " pixels", "application-resize");
  }

  // Restart the rendering when when the camera, lights and render params are dirty
  if (m_renderSettings->m_DirtyFlags.HasFlag(CameraDirty | LightsDirty | RenderParamsDirty | TransferFunctionDirty |
                                             RoiDirty)) {
    if (m_renderSettings->m_DirtyFlags.HasFlag(TransferFunctionDirty)) {
      // TODO: only update the ones that changed.
      int NC = m_scene->m_volume->sizeC();
      for (int i = 0; i < NC; ++i) {
        m_imgCuda->updateLutGpu(i, m_scene->m_volume.get());
      }
    }

    //		ResetRenderCanvasView();

    // Reset no. iterations
    m_renderSettings->SetNoIterations(0);
  }
  if (m_renderSettings->m_DirtyFlags.HasFlag(LightsDirty)) {
    for (int i = 0; i < m_scene->m_lighting.m_NoLights; ++i) {
      m_scene->m_lighting.m_Lights[i].Update(m_scene->m_boundingBox);
    }

    // Reset no. iterations
    m_renderSettings->SetNoIterations(0);
  }
  if (m_renderSettings->m_DirtyFlags.HasFlag(VolumeDataDirty)) {
    int ch[4] = { 0, 0, 0, 0 };
    int activeChannel = 0;
    int NC = m_scene->m_volume->sizeC();
    for (int i = 0; i < NC; ++i) {
      if (m_scene->m_material.m_enabled[i] && activeChannel < 4) {
        ch[activeChannel] = i;
        activeChannel++;
      }
    }
    m_imgCuda->updateVolumeData4x16(m_scene->m_volume.get(), ch[0], ch[1], ch[2], ch[3]);
    m_renderSettings->SetNoIterations(0);
  }
  // At this point, all dirty flags should have been taken care of, since the flags in the original scene are now
  // cleared
  m_renderSettings->m_DirtyFlags.ClearAllFlags();

  m_renderSettings->m_RenderSettings.m_GradientDelta = 1.0f / (float)this->m_scene->m_volume->maxPixelDimension();

  m_renderSettings->m_DenoiseParams.SetWindowRadius(3.0f);

  CudaLighting cudalt;
  FillCudaLighting(m_scene, cudalt);
  CudaCamera cudacam;
  FillCudaCamera(&(camera), cudacam);

  glm::vec3 sn = m_scene->m_boundingBox.GetMinP();
  glm::vec3 ext = m_scene->m_boundingBox.GetExtent();
  CBoundingBox b;
  b.SetMinP(glm::vec3(ext.x * m_scene->m_roi.GetMinP().x + sn.x,
                      ext.y * m_scene->m_roi.GetMinP().y + sn.y,
                      ext.z * m_scene->m_roi.GetMinP().z + sn.z));
  b.SetMaxP(glm::vec3(ext.x * m_scene->m_roi.GetMaxP().x + sn.x,
                      ext.y * m_scene->m_roi.GetMaxP().y + sn.y,
                      ext.z * m_scene->m_roi.GetMaxP().z + sn.z));

  cudaBoundingBox clipbb;
  clipbb.m_min = make_float3(b.GetMinP().x, b.GetMinP().y, b.GetMinP().z);
  clipbb.m_max = make_float3(b.GetMaxP().x, b.GetMaxP().y, b.GetMaxP().z);
  cudaBoundingBox aabb;
  aabb.m_min = make_float3(
    m_scene->m_boundingBox.GetMinP().x, m_scene->m_boundingBox.GetMinP().y, m_scene->m_boundingBox.GetMinP().z);
  aabb.m_max = make_float3(
    m_scene->m_boundingBox.GetMaxP().x, m_scene->m_boundingBox.GetMaxP().y, m_scene->m_boundingBox.GetMaxP().z);

  BindConstants(cudalt,
                m_renderSettings->m_DenoiseParams,
                cudacam,
                aabb,
                clipbb,
                m_renderSettings->m_RenderSettings,
                m_renderSettings->GetNoIterations(),
                m_w,
                m_h,
                camera.m_Film.m_Gamma,
                camera.m_Film.m_Exposure);
  // Render image
  // RayMarchVolume(_cudaF32Buffer, _volumeTex, _volumeGradientTex, _renderSettings, _w, _h, 2.0f, 20.0f,
  // glm::value_ptr(m), _channelMin, _channelMax);
  cudaFB theCudaFB = { m_cudaF32Buffer, m_cudaF32AccumBuffer, m_randomSeeds1, m_randomSeeds2 };

  // single channel
  int NC = m_scene->m_volume->sizeC();
  // use first 3 channels only.
  int activeChannel = 0;
  cudaVolume theCudaVolume(0);
  for (int i = 0; i < NC; ++i) {
    if (m_scene->m_material.m_enabled[i] && activeChannel < MAX_CUDA_CHANNELS) {
      theCudaVolume.m_volumeTexture[activeChannel] = m_imgCuda->m_volumeTextureInterleaved;
      theCudaVolume.m_gradientVolumeTexture[activeChannel] = m_imgCuda->m_channels[i].m_volumeGradientTexture;
      theCudaVolume.m_lutTexture[activeChannel] = m_imgCuda->m_channels[i].m_volumeLutTexture;
      theCudaVolume.m_intensityMax[activeChannel] = m_scene->m_volume->channel(i)->m_max;
      theCudaVolume.m_intensityMin[activeChannel] = m_scene->m_volume->channel(i)->m_min;
      theCudaVolume.m_diffuse[activeChannel * 3 + 0] = m_scene->m_material.m_diffuse[i * 3 + 0];
      theCudaVolume.m_diffuse[activeChannel * 3 + 1] = m_scene->m_material.m_diffuse[i * 3 + 1];
      theCudaVolume.m_diffuse[activeChannel * 3 + 2] = m_scene->m_material.m_diffuse[i * 3 + 2];
      theCudaVolume.m_specular[activeChannel * 3 + 0] = m_scene->m_material.m_specular[i * 3 + 0];
      theCudaVolume.m_specular[activeChannel * 3 + 1] = m_scene->m_material.m_specular[i * 3 + 1];
      theCudaVolume.m_specular[activeChannel * 3 + 2] = m_scene->m_material.m_specular[i * 3 + 2];
      theCudaVolume.m_emissive[activeChannel * 3 + 0] = m_scene->m_material.m_emissive[i * 3 + 0];
      theCudaVolume.m_emissive[activeChannel * 3 + 1] = m_scene->m_material.m_emissive[i * 3 + 1];
      theCudaVolume.m_emissive[activeChannel * 3 + 2] = m_scene->m_material.m_emissive[i * 3 + 2];
      theCudaVolume.m_roughness[activeChannel] = m_scene->m_material.m_roughness[i];
      theCudaVolume.m_opacity[activeChannel] = m_scene->m_material.m_opacity[i];

      activeChannel++;
      theCudaVolume.m_nChannels = activeChannel;
    }
  }

  // find nearest intersection to set camera focal distance automatically.
  // then re-upload that data.
  if (camera.m_Focus.m_Type == 0) {
    ComputeFocusDistance(theCudaVolume);
  }

  int numIterations = m_renderSettings->GetNoIterations();
  Render(0,
         camera.m_Film.m_ExposureIterations,
         camera.m_Film.m_Resolution.GetResX(),
         camera.m_Film.m_Resolution.GetResY(),
         theCudaFB,
         theCudaVolume,
         m_timingRender,
         m_timingBlur,
         m_timingPostProcess,
         m_timingDenoise,
         numIterations);
  m_renderSettings->SetNoIterations(numIterations);
  // LOG_DEBUG << "RETURN FROM RENDER";

  // Tonemap into opengl display buffer

  // do cuda with cudaSurfaceObj

  // set the lerpC here because the Render call is incrementing the number of iterations.
  //_renderSettings->m_DenoiseParams.m_LerpC = 0.33f * (max((float)_renderSettings->GetNoIterations(), 1.0f)
  //* 1.0f);//1.0f - powf(1.0f / (float)gScene.GetNoIterations(), 15.0f);//1.0f - expf(-0.01f *
  //(float)gScene.GetNoIterations());
  m_renderSettings->m_DenoiseParams.m_LerpC =
    0.33f * (max((float)m_renderSettings->GetNoIterations(), 1.0f) *
             0.035f); // 1.0f - powf(1.0f / (float)gScene.GetNoIterations(), 15.0f);//1.0f - expf(-0.01f *
                      // (float)gScene.GetNoIterations());
  //	LOG_DEBUG << "Window " << _w << " " << _h << " Cam " << _renderSettings->m_Camera.m_Film.m_Resolution.GetResX()
  //<< " " << _renderSettings->m_Camera.m_Film.m_Resolution.GetResY();
  CCudaTimer TmrDenoise;
  if (m_renderSettings->m_DenoiseParams.m_Enabled && m_renderSettings->m_DenoiseParams.m_LerpC > 0.0f &&
      m_renderSettings->m_DenoiseParams.m_LerpC < 1.0f) {
    Denoise(m_cudaF32AccumBuffer, m_cudaGLSurfaceObject, m_w, m_h, m_renderSettings->m_DenoiseParams.m_LerpC);
  } else {
    ToneMap(m_cudaF32AccumBuffer, m_cudaGLSurfaceObject, m_w, m_h);
  }
  m_timingDenoise.AddDuration(TmrDenoise.ElapsedTime());

  HandleCudaError(cudaStreamSynchronize(0));

  // display timings.

  m_status.SetStatisticChanged(
    "Performance", "Render Image", QString::number(m_timingRender.m_FilteredDuration, 'f', 2), "ms");
  m_status.SetStatisticChanged(
    "Performance", "Blur Estimate", QString::number(m_timingBlur.m_FilteredDuration, 'f', 2), "ms");
  m_status.SetStatisticChanged(
    "Performance", "Post Process Estimate", QString::number(m_timingPostProcess.m_FilteredDuration, 'f', 2), "ms");
  m_status.SetStatisticChanged(
    "Performance", "De-noise Image", QString::number(m_timingDenoise.m_FilteredDuration, 'f', 2), "ms");

  // FPS.AddDuration(1000.0f / TmrFps.ElapsedTime());

  //_status.SetStatisticChanged("Performance", "FPS", QString::number(FPS.m_FilteredDuration, 'f', 2), "Frames/Sec.");
  m_status.SetStatisticChanged(
    "Performance", "No. Iterations", QString::number(m_renderSettings->GetNoIterations()), "");
}

void
RenderGLCuda::render(const CCamera& camera)
{
  // draw to _fbtex
  doRender(camera);

  // put _fbtex to main render target
  drawImage();
}

void
RenderGLCuda::drawImage()
{
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // draw quad using the tex that cudaTex was mapped to
  m_imagequad->draw(m_fbtex);
}

void
RenderGLCuda::resize(uint32_t w, uint32_t h)
{
  // w = 8; h = 8;
  glViewport(0, 0, w, h);
  if ((m_w == w) && (m_h == h)) {
    return;
  }

  initFB(w, h);
  LOG_DEBUG << "Resized window to " << w << " x " << h;

  m_w = w;
  m_h = h;
}

void
RenderGLCuda::cleanUpResources()
{

  delete m_imagequad;
  m_imagequad = nullptr;

  cleanUpFB();
}

RenderParams&
RenderGLCuda::renderParams()
{
  return m_renderParams;
}

Scene*
RenderGLCuda::scene()
{
  return m_scene;
}

void
RenderGLCuda::setScene(Scene* s)
{
  m_scene = s;
}

size_t
RenderGLCuda::getGpuBytes()
{
  return m_gpuBytes + m_imgCuda->m_gpuBytes;
}
