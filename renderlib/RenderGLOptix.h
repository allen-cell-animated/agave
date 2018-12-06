#pragma once
#include "IRenderWindow.h"

#include "AppScene.h"
#include "RenderSettings.h"

#include "Status.h"
#include "Timing.h"
#include "glad/include/glad/glad.h"

#include <optix.h>
#include <optixu/optixpp_namespace.h>
//#include <optixu/optixpp.h>

#include <memory>

class ImageXYZC;
class RectImage2D;
struct OptiXMesh;

class RenderGLOptix : public IRenderWindow
{
public:
  RenderGLOptix(RenderSettings* rs);
  virtual ~RenderGLOptix();

  virtual void initialize(uint32_t w, uint32_t h);
  virtual void render(const CCamera& camera);
  virtual void resize(uint32_t w, uint32_t h);
  virtual void cleanUpResources();
  virtual RenderParams& renderParams();
  virtual Scene* scene();
  virtual void setScene(Scene* s);

  virtual CStatus* getStatusInterface() { return &m_status; }

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

  int m_w, m_h;

  RectImage2D* m_imagequad;

  CStatus m_status;

  size_t m_gpuBytes;

  optix::Context m_ctx;
  RTcontext m_context;

  /* Primary RTAPI objects */
  RTprogram m_ray_gen_program;
  RTprogram m_miss_program;
  RTprogram m_exception_program;
  RTbuffer m_buffer;

  optix::Program m_phong_closesthit_program;
  optix::Program m_phong_anyhit_program;
  optix::Program m_mesh_intersect_program;
  optix::Program m_mesh_boundingbox_program;

  /* Parameters */
  RTvariable m_result_buffer;
  RTvariable m_draw_color;
  RTvariable m_scene_epsilon;
  RTvariable m_eye;
  RTvariable m_U;
  RTvariable m_V;
  RTvariable m_W;

  RTvariable m_lightsvar;

  RTbuffer m_light_buffer;

  void initOptixMesh();
  // the scene root node...
  RTgroup m_topGroup;

  std::vector<std::shared_ptr<OptiXMesh>> m_optixmeshes;
};
