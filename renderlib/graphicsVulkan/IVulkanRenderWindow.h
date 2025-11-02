#pragma once

#include <inttypes.h>

class CCamera;
class CStatus;
class VulkanFramebufferObject;
class RenderSettings;
class Scene;

#include <memory>

// Vulkan version of IRenderWindow - exact same interface but Vulkan implementation
class IVulkanRenderWindow
{
public:
  IVulkanRenderWindow();
  virtual ~IVulkanRenderWindow();

  virtual void initialize(uint32_t w, uint32_t h) = 0;
  virtual void render(const CCamera& camera) = 0;
  virtual void renderTo(const CCamera& camera, VulkanFramebufferObject* fbo) = 0;
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