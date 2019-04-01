#include "RenderGLPT.h"

#include "glad/glad.h"
#include "glm.h"

#include "ImageXYZC.h"
#include "Logging.h"
#include "gl/Util.h"
#include "gl/v33/V33FSQ.h"
#include "gl/v33/V33Image3D.h"
#include "glsl/v330/GLCopyShader.h"
#include "glsl/v330/GLPTVolumeShader.h"
#include "glsl/v330/GLToneMapShader.h"
#include "glsl/v330/V330GLImageShader2DnoLut.h"

//#include "cudarndr/Camera2.cuh"
//#include "cudarndr/Lighting2.cuh"

#include <array>

RenderGLPT::RenderGLPT(RenderSettings* rs)
  : m_glF32Buffer(0)
  , m_glF32AccumBuffer(0)
  , m_fbF32(0)
  , m_fbF32Accum(0)
  , m_fbtex(0)
  , m_renderBufferShader(nullptr)
  , m_copyShader(nullptr)
  , m_toneMapShader(nullptr)
  , m_fsq(nullptr)
  , m_randomSeeds1(nullptr)
  , m_randomSeeds2(nullptr)
  , m_renderSettings(rs)
  , m_w(0)
  , m_h(0)
  , m_scene(nullptr)
  , m_gpuBytes(0)
  , m_imagequad(nullptr)
  , m_RandSeed(0)
{}

RenderGLPT::~RenderGLPT() {}

void
RenderGLPT::cleanUpFB()
{
  // destroy the framebuffer texture
  if (m_fbtex) {
    glBindTexture(GL_TEXTURE_2D, 0);
    glDeleteTextures(1, &m_fbtex);
    check_gl("Destroy fb texture");
    m_fbtex = 0;
  }
  if (m_fb) {
    glDeleteFramebuffers(1, &m_fb);
    m_fb = 0;
  }

  if (m_fbF32) {
    glDeleteFramebuffers(1, &m_fbF32);
    m_fbF32 = 0;
  }
  if (m_glF32Buffer) {
    glBindTexture(GL_TEXTURE_2D, 0);
    glDeleteTextures(1, &m_glF32Buffer);
    check_gl("Destroy fb texture");
    m_glF32Buffer = 0;
  }
  if (m_fbF32Accum) {
    glDeleteFramebuffers(1, &m_fbF32Accum);
    m_fbF32Accum = 0;
  }
  if (m_glF32AccumBuffer) {
    glBindTexture(GL_TEXTURE_2D, 0);
    glDeleteTextures(1, &m_glF32AccumBuffer);
    check_gl("Destroy fb texture");
    m_glF32AccumBuffer = 0;
  }

  delete m_renderBufferShader;
  m_renderBufferShader = 0;
  delete m_copyShader;
  m_copyShader = 0;
  delete m_toneMapShader;
  m_toneMapShader = 0;
  delete m_fsq;
  m_fsq = 0;

  m_gpuBytes = 0;
}

void
RenderGLPT::initFB(uint32_t w, uint32_t h)
{
  cleanUpFB();

  glGenTextures(1, &m_glF32Buffer);
  check_gl("Gen fb texture id");
  glBindTexture(GL_TEXTURE_2D, m_glF32Buffer);
  check_gl("Bind fb texture");
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  m_gpuBytes += w * h * 4 * sizeof(float);
  check_gl("Create fb texture");

  glGenFramebuffers(1, &m_fbF32);
  glBindFramebuffer(GL_FRAMEBUFFER, m_fbF32);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_glF32Buffer, 0);
  check_glfb("resized float pathtrace sample fb");

  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT);

  glGenTextures(1, &m_glF32AccumBuffer);
  check_gl("Gen fb texture id");
  glBindTexture(GL_TEXTURE_2D, m_glF32AccumBuffer);
  check_gl("Bind fb texture");
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0, GL_RGBA, GL_FLOAT, 0);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  m_gpuBytes += w * h * 4 * sizeof(float);
  check_gl("Create fb texture");

  glGenFramebuffers(1, &m_fbF32Accum);
  glBindFramebuffer(GL_FRAMEBUFFER, m_fbF32Accum);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_glF32AccumBuffer, 0);
  check_glfb("resized float accumulation fb");

  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT);

  m_fsq = new FSQ();
  m_fsq->setSize(glm::vec2(-1, 1), glm::vec2(-1, 1));
  m_fsq->create();
  m_renderBufferShader = new GLPTVolumeShader();
  m_copyShader = new GLCopyShader();
  m_toneMapShader = new GLToneMapShader();

  {
    unsigned int* pSeeds = (unsigned int*)malloc(w * h * sizeof(unsigned int));
    memset(pSeeds, 0, w * h * sizeof(unsigned int));
    for (unsigned int i = 0; i < w * h; i++)
      pSeeds[i] = rand();
    // m_gpuBytes += w * h * sizeof(unsigned int);
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

  glGenFramebuffers(1, &m_fb);
  glBindFramebuffer(GL_FRAMEBUFFER, m_fb);
  glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, m_fbtex, 0);
  check_glfb("resized main fb");

  // clear this fb to black
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT);
}

void
RenderGLPT::initVolumeTextureCUDA()
{
  // free the gpu resources of the old image.
  m_imgCuda.deallocGpu();

  if (!m_scene || !m_scene->m_volume) {
    return;
  }
  //    ImageCuda cimg;
  //    cimg.allocGpuInterleaved(m_scene->m_volume.get());
  //    m_imgCuda = cimg;
  m_imgCuda.allocGpuInterleaved(m_scene->m_volume.get());
}

void
RenderGLPT::initialize(uint32_t w, uint32_t h)
{
  m_imagequad = new RectImage2D();

  initVolumeTextureCUDA();
  check_gl("init gl volume");

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  check_gl("init gl state");

  // Size viewport
  resize(w, h);
}

void
RenderGLPT::doRender(const CCamera& camera)
{
  if (!m_scene || !m_scene->m_volume) {
    return;
  }

  GLint drawFboId = 0;
  glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &drawFboId);

  if (!m_imgCuda.m_VolumeGLTexture || m_renderSettings->m_DirtyFlags.HasFlag(VolumeDirty)) {
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
        m_imgCuda.updateLutGpu(i, m_scene->m_volume.get());
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
    m_imgCuda.updateVolumeData4x16(m_scene->m_volume.get(), ch[0], ch[1], ch[2], ch[3]);
    m_renderSettings->SetNoIterations(0);
  }
  // At this point, all dirty flags should have been taken care of, since the flags in the original scene are now
  // cleared
  m_renderSettings->m_DirtyFlags.ClearAllFlags();

  m_renderSettings->m_RenderSettings.m_GradientDelta = 1.0f / (float)this->m_scene->m_volume->maxPixelDimension();

  m_renderSettings->m_DenoiseParams.SetWindowRadius(3.0f);

  // CudaLighting cudalt;
  // FillCudaLighting(m_scene, cudalt);
  // CudaCamera cudacam;
  // FillCudaCamera(&(camera), cudacam);

  glm::vec3 sn = m_scene->m_boundingBox.GetMinP();
  glm::vec3 ext = m_scene->m_boundingBox.GetExtent();
  CBoundingBox b;
  b.SetMinP(glm::vec3(ext.x * m_scene->m_roi.GetMinP().x + sn.x,
                      ext.y * m_scene->m_roi.GetMinP().y + sn.y,
                      ext.z * m_scene->m_roi.GetMinP().z + sn.z));
  b.SetMaxP(glm::vec3(ext.x * m_scene->m_roi.GetMaxP().x + sn.x,
                      ext.y * m_scene->m_roi.GetMaxP().y + sn.y,
                      ext.z * m_scene->m_roi.GetMaxP().z + sn.z));

  int numIterations = m_renderSettings->GetNoIterations();

  glm::mat4 m(1.0);

  GLuint accumTargetTex = m_glF32Buffer;          // the texture of m_fbF32
  GLuint prevAccumTargetTex = m_glF32AccumBuffer; // the texture that will be tonemapped to screen, a copy of m_fbF32

  glDisable(GL_DEPTH_TEST);
  glDisable(GL_BLEND);

  for (int i = 0; i < camera.m_Film.m_ExposureIterations; ++i) {
    GLTimer TmrRender;

    // 1. draw pathtrace pass and accumulate, using prevAccumTargetTex as previous accumulation
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbF32);
    check_glfb("bind framebuffer for pathtrace iteration");

    m_renderBufferShader->bind();

    m_renderBufferShader->setShadingUniforms(m_scene,
                                             m_renderSettings->m_DenoiseParams,
                                             camera,
                                             b,
                                             m_renderSettings->m_RenderSettings,
                                             numIterations,
                                             m_RandSeed,
                                             m_w,
                                             m_h,
                                             m_imgCuda,
                                             prevAccumTargetTex);

    m_fsq->render(m);

    m_timingRender.AddDuration(TmrRender.ElapsedTime());

    // unbind the prevAccumTargetTex
    glActiveTexture(GL_TEXTURE0 + 1);
    glBindTexture(GL_TEXTURE_2D, 0);

    // 2. copy to accumTargetTex texture that will be used as accumulator for next pass

    glBindFramebuffer(GL_FRAMEBUFFER, m_fbF32Accum);
    check_glfb("bind framebuffer for accumulator");

    // the sample
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_glF32Buffer);

    m_copyShader->bind();
    m_copyShader->setShadingUniforms();

    m_fsq->render(m);

    m_copyShader->release();
    //_timingPostProcess.AddDuration(TmrPostProcess.ElapsedTime());

    // ping pong accum buffer. this will stall till previous accum render is done.

    numIterations++;
    m_RandSeed++;
  }

  m_renderSettings->SetNoIterations(numIterations);

  // TODO do denoising in GLSL

  // set the lerpC here because the Render call is incrementing the number of iterations.
  // m_renderSettings->m_DenoiseParams.m_LerpC = 0.33f * (max((float)m_renderSettings->GetNoIterations(), 1.0f)
  // * 1.0f);//1.0f - powf(1.0f / (float)gScene.GetNoIterations(), 15.0f);//1.0f - expf(-0.01f *
  // (float)gScene.GetNoIterations());
  m_renderSettings->m_DenoiseParams.m_LerpC =
    0.33f * (std::max((float)m_renderSettings->GetNoIterations(), 1.0f) * 0.035f);
  // 1.0f - powf(1.0f / (float)gScene.GetNoIterations(), 15.0f);//1.0f - expf(-0.01f *
  // (float)gScene.GetNoIterations());
  //	LOG_DEBUG << "Window " << _w << " " << _h << " Cam " << m_renderSettings->m_Camera.m_Film.m_Resolution.GetResX()
  //<< " " << m_renderSettings->m_Camera.m_Film.m_Resolution.GetResY(); CCudaTimer TmrDenoise;
  if (m_renderSettings->m_DenoiseParams.m_Enabled && m_renderSettings->m_DenoiseParams.m_LerpC > 0.0f &&
      m_renderSettings->m_DenoiseParams.m_LerpC < 1.0f) {
    // draw from accum buffer into fbtex
    // Denoise(_cudaF32AccumBuffer, _fbTex, _w, _h, m_renderSettings->m_DenoiseParams.m_LerpC);
  } else {
    // ToneMap(_cudaF32AccumBuffer, _fbTex, _w, _h);
  }
  //_timingDenoise.AddDuration(TmrDenoise.ElapsedTime());

  // Tonemap into opengl display buffer
  glBindFramebuffer(GL_FRAMEBUFFER, m_fb);
  check_glfb("bind framebuffer for tone map");

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, m_glF32AccumBuffer);

  m_toneMapShader->bind();
  m_toneMapShader->setShadingUniforms(1.0f / camera.m_Film.m_Exposure);

  m_fsq->render(m);

  m_toneMapShader->release();

  // LOG_DEBUG << "RETURN FROM RENDER";

  // display timings.

  m_status.SetStatisticChanged(
    "Performance", "Render Image", QString::number(m_timingRender.m_FilteredDuration, 'f', 2), "ms.");
  m_status.SetStatisticChanged(
    "Performance", "Blur Estimate", QString::number(m_timingBlur.m_FilteredDuration, 'f', 2), "ms.");
  m_status.SetStatisticChanged(
    "Performance", "Post Process Estimate", QString::number(m_timingPostProcess.m_FilteredDuration, 'f', 2), "ms.");
  m_status.SetStatisticChanged(
    "Performance", "De-noise Image", QString::number(m_timingDenoise.m_FilteredDuration, 'f', 2), "ms.");

  // FPS.AddDuration(1000.0f / TmrFps.ElapsedTime());

  // m_status.SetStatisticChanged("Performance", "FPS", QString::number(FPS.m_FilteredDuration, 'f', 2), "Frames/Sec.");
  m_status.SetStatisticChanged(
    "Performance", "No. Iterations", QString::number(m_renderSettings->GetNoIterations()), "Iterations");

  glBindFramebuffer(GL_FRAMEBUFFER, drawFboId);
  check_glfb("bind framebuffer for final draw");

  glEnable(GL_BLEND);
  glEnable(GL_DEPTH_TEST);
}

void
RenderGLPT::render(const CCamera& camera)
{
  // draw to m_fbtex
  doRender(camera);

  // put m_fbtex to main render target
  drawImage();
}

void
RenderGLPT::drawImage()
{
  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // draw quad using the tex that cudaTex was mapped to
  m_imagequad->draw(m_fbtex);
}

void
RenderGLPT::resize(uint32_t w, uint32_t h)
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

  m_renderSettings->SetNoIterations(0);
}

void
RenderGLPT::cleanUpResources()
{
  m_imgCuda.deallocGpu();

  delete m_imagequad;
  m_imagequad = nullptr;

  cleanUpFB();
}

RenderParams&
RenderGLPT::renderParams()
{
  return m_renderParams;
}

Scene*
RenderGLPT::scene()
{
  return m_scene;
}

void
RenderGLPT::setScene(Scene* s)
{
  m_scene = s;
}

size_t
RenderGLPT::getGpuBytes()
{
  return m_gpuBytes + m_imgCuda.m_gpuBytes;
}
