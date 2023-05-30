#pragma once
#include "AppScene.h"
#include "IRenderWindow.h"
#include "Status.h"
#include "Timing.h"

#include <chrono>
#include <memory>

class Image3D;
class ImageXYZC;
class RenderSettings;

class RenderGL : public IRenderWindow
{
public:
  RenderGL(RenderSettings* rs);
  virtual ~RenderGL();

  virtual void initialize(uint32_t w, uint32_t h, float devicePixelRatio = 1.0f);
  virtual void render(const CCamera& camera);
  virtual void renderTo(const CCamera& camera, GLFramebufferObject* fbo);
  virtual void resize(uint32_t w, uint32_t h, float devicePixelRatio = 1.0f);
  virtual void cleanUpResources();

  virtual std::shared_ptr<CStatus> getStatusInterface() { return m_status; }
  virtual RenderParams& renderParams();
  virtual Scene* scene();
  virtual void setScene(Scene* s);

  Image3D* getImage() const { return m_image3d; };

private:
  Image3D* m_image3d;
  RenderSettings* m_renderSettings;

  Scene* m_scene;
  RenderParams m_renderParams;

  std::shared_ptr<CStatus> m_status;
  Timing m_timingRender;
  std::chrono::time_point<std::chrono::high_resolution_clock> mStartTime;

  int m_w, m_h;
  float m_devicePixelRatio;

  void initFromScene();
  bool prepareToRender();
  void doClear();
};
