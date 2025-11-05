#pragma once

#include "graphics/IRenderWindowBase.h"
#include <inttypes.h>

class CCamera;
class CStatus;
class VulkanFramebufferObject;
class RenderSettings;
class Scene;

#include <memory>

// Vulkan version of IRenderWindow - exact same interface but Vulkan implementation
class IVulkanRenderWindow : public IRenderWindowBase
{
public:
  IVulkanRenderWindow();
  virtual ~IVulkanRenderWindow();

  // Vulkan-specific version of renderTo
  virtual void renderTo(const CCamera& camera, VulkanFramebufferObject* fbo) = 0;
};