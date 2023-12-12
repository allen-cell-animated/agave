#pragma once

#include "../renderlib/graphics/IRenderWindow.h"

class Scene;
class CStatus;
class RenderSettings;
class BoundingBoxDrawable;

class RenderWgpuPT : public IRenderWindow
{
public:
  RenderWgpuPT(RenderSettings* rs);
  virtual ~RenderWgpuPT();

  virtual void initialize(uint32_t w, uint32_t h);
  virtual void render(const CCamera& camera);
  virtual void renderTo(const CCamera& camera, GLFramebufferObject* fbo);
  virtual void resize(uint32_t w, uint32_t h);
  virtual void getSize(uint32_t& w, uint32_t& h)
  {
    w = m_w;
    h = m_h;
  }
  virtual void cleanUpResources() {}
  virtual RenderSettings& renderSettings() { return *m_renderSettings; }
  virtual Scene* scene() { return m_scene; }
  virtual void setScene(Scene* s) { m_scene = s; }

  virtual std::shared_ptr<CStatus> getStatusInterface() { return m_status; }

private:
  RenderSettings* m_renderSettings;
  Scene* m_scene;

  BoundingBoxDrawable* m_boundingBoxDrawable;

  // screen size auxiliary buffers for rendering
  unsigned int* m_randomSeeds1;
  unsigned int* m_randomSeeds2;
  // incrementing integer to give to shader
  int m_RandSeed;

  int m_w, m_h;

  std::shared_ptr<CStatus> m_status;
};
