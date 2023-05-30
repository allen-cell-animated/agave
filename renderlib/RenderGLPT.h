#pragma once
#include "IRenderWindow.h"

#include <glad/glad.h>

#include "AppScene.h"
#include "RenderSettings.h"

#include "ImageXyzcGpu.h"
#include "Status.h"
#include "Timing.h"

#include <memory>

class BoundingBoxDrawable;
class Framebuffer;
class FSQ;
class ImageXYZC;
class Image3D;
class RectImage2D;
class GLCopyShader;
class GLPTVolumeShader;
class GLToneMapShader;

class RenderGLPT : public IRenderWindow
{
public:
  RenderGLPT(RenderSettings* rs);
  virtual ~RenderGLPT();

  virtual void initialize(uint32_t w, uint32_t h, float devicePixelRatio = 1.0f);
  virtual void render(const CCamera& camera);
  virtual void renderTo(const CCamera& camera, GLFramebufferObject* fbo);
  virtual void resize(uint32_t w, uint32_t h, float devicePixelRatio = 1.0f);
  virtual void cleanUpResources();
  virtual RenderParams& renderParams();
  virtual Scene* scene();
  virtual void setScene(Scene* s);

  virtual std::shared_ptr<CStatus> getStatusInterface() { return m_status; }

  Image3D* getImage() const { return nullptr; };
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
  void initVolumeTextureGpu();
  void cleanUpFB();

  ImageGpu m_imgGpu;

  RectImage2D* m_imagequad;

  // the rgba8 buffer for display
  Framebuffer* m_fb;

  FSQ* m_fsq;

  // the rgbaf32 buffer for rendering
  Framebuffer* m_fbF32;
  GLPTVolumeShader* m_renderBufferShader;

  // the rgbaf32 accumulation buffer that holds the progressively rendered image
  Framebuffer* m_fbF32Accum;
  GLCopyShader* m_copyShader;
  GLToneMapShader* m_toneMapShader;

  BoundingBoxDrawable* m_boundingBoxDrawable;

  // screen size auxiliary buffers for rendering
  unsigned int* m_randomSeeds1;
  unsigned int* m_randomSeeds2;
  // incrementing integer to give to shader
  int m_RandSeed;

  int m_w, m_h;
  float m_devicePixelRatio;

  Timing m_timingRender, m_timingBlur, m_timingPostProcess, m_timingDenoise;
  std::shared_ptr<CStatus> m_status;

  size_t m_gpuBytes;
};
