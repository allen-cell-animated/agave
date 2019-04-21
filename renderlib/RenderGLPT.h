#pragma once
#include "IRenderWindow.h"

#include <glad/glad.h>

#include "AppScene.h"
#include "RenderSettings.h"

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
  virtual void resize(uint32_t w, uint32_t h, float devicePixelRatio = 1.0f);
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

  ImageCuda m_imgCuda;

  RectImage2D* m_imagequad;

  // the rgba8 buffer for display
  GLuint m_fbtex;
  GLuint m_fb;

  FSQ* m_fsq;

  // the rgbaf32 buffer for rendering
  GLuint m_glF32Buffer;
  GLuint m_fbF32;
  GLPTVolumeShader* m_renderBufferShader;

  // the rgbaf32 accumulation buffer that holds the progressively rendered image
  GLuint m_glF32AccumBuffer;
  GLuint m_fbF32Accum;
  GLCopyShader* m_copyShader;
  GLToneMapShader* m_toneMapShader;

  // screen size auxiliary buffers for rendering
  unsigned int* m_randomSeeds1;
  unsigned int* m_randomSeeds2;
  // incrementing integer to give to shader
  int m_RandSeed;

  int m_w, m_h;
  float m_devicePixelRatio;

  Timing m_timingRender, m_timingBlur, m_timingPostProcess, m_timingDenoise;
  CStatus m_status;

  size_t m_gpuBytes;

  void FillCudaLighting(Scene* pScene, CudaLighting& cl);
  void FillCudaCamera(const CCamera* pCamera, CudaCamera& c);
};
