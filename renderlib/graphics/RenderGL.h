#pragma once
#include "AppScene.h"
#include "IRenderWindow.h"
#include "Status.h"
#include "Timing.h"

#include <chrono>
#include <memory>

class BoundingBoxDrawable;
class Image3D;
class ImageXYZC;
class RenderSettings;

class RenderGL : public IRenderWindow
{
public:
  static const std::string TYPE_NAME;
  RenderGL(RenderSettings* rs);
  virtual ~RenderGL();

  virtual void initialize(uint32_t w, uint32_t h);
  virtual void render(const CCamera& camera);
  virtual void renderTo(const CCamera& camera, GLFramebufferObject* fbo);
  virtual void resize(uint32_t w, uint32_t h);
  virtual void getSize(uint32_t& w, uint32_t& h)
  {
    w = m_w;
    h = m_h;
  }
  virtual void cleanUpResources();

  virtual std::shared_ptr<CStatus> getStatusInterface() { return m_status; }
  virtual RenderSettings& renderSettings();
  virtual Scene* scene();
  virtual void setScene(Scene* s);

  Image3D* getImage() const { return m_image3d; };

private:
  Image3D* m_image3d;
  BoundingBoxDrawable* m_boundingBoxDrawable;
  RenderSettings* m_renderSettings;

  Scene* m_scene;

  std::shared_ptr<CStatus> m_status;
  Timing m_timingRender;
  std::chrono::time_point<std::chrono::high_resolution_clock> mStartTime;

  int m_w, m_h;

  void initFromScene();
  bool prepareToRender();
  void doClear();
  void drawSceneObjects(const CCamera& camera);
};
