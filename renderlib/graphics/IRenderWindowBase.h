#pragma once

#include <inttypes.h>
#include <memory>

class CCamera;
class CStatus;
class RenderSettings;
class Scene;

// Common base interface for both OpenGL and Vulkan render windows
class IRenderWindowBase
{
public:
  IRenderWindowBase();
  virtual ~IRenderWindowBase();

  virtual void initialize(uint32_t w, uint32_t h) = 0;
  virtual void render(const CCamera& camera) = 0;
  virtual void resize(uint32_t w, uint32_t h) = 0;
  virtual void getSize(uint32_t& w, uint32_t& h) = 0;
  virtual void cleanUpResources() {}

  // An interface for reporting statistics and other data updates
  virtual std::shared_ptr<CStatus> getStatusInterface() { return nullptr; }

  // I own these.
  virtual RenderSettings& renderSettings() = 0;

  virtual Scene* scene() = 0;
  virtual void setScene(Scene* s) = 0;
};