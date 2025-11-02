#pragma once

#include "AppScene.h"
#include "IVulkanRenderWindow.h"
#include "Status.h"
#include "Timing.h"

#include <vulkan/vulkan.h>
#include <chrono>
#include <memory>
#include <vector>

class BoundingBoxDrawableVK;
class Image3DVK;
class ImageXYZC;
class RenderSettings;
class VulkanDevice;
class VulkanSwapchain;
class VulkanCommandBuffer;

class RenderVK : public IVulkanRenderWindow
{
public:
  static const std::string TYPE_NAME;
  RenderVK(RenderSettings* rs);
  virtual ~RenderVK();

  virtual void initialize(uint32_t w, uint32_t h);
  virtual void render(const CCamera& camera);
  virtual void renderTo(const CCamera& camera, VulkanFramebufferObject* fbo);
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

  Image3DVK* getImage() const { return m_image3d; }

private:
  // Vulkan objects
  VkInstance m_instance;
  VkPhysicalDevice m_physicalDevice;
  VkDevice m_device;
  VkQueue m_graphicsQueue;
  VkQueue m_presentQueue;
  VkSurfaceKHR m_surface;
  VkRenderPass m_renderPass;
  VkCommandPool m_commandPool;
  std::vector<VkCommandBuffer> m_commandBuffers;

  // Synchronization objects
  std::vector<VkSemaphore> m_imageAvailableSemaphores;
  std::vector<VkSemaphore> m_renderFinishedSemaphores;
  std::vector<VkFence> m_inFlightFences;

  // Application objects
  Image3DVK* m_image3d;
  RenderSettings* m_renderSettings;
  Scene* m_scene;
  std::shared_ptr<CStatus> m_status;
  Timing m_timingRender;
  std::chrono::time_point<std::chrono::high_resolution_clock> mStartTime;

  int m_w, m_h;
  uint32_t m_currentFrame;

  // Private methods
  bool initVulkan();
  bool createInstance();
  bool setupDebugMessenger();
  bool createSurface();
  bool pickPhysicalDevice();
  bool createLogicalDevice();
  bool createSwapChain();
  bool createImageViews();
  bool createRenderPass();
  bool createDescriptorSetLayout();
  bool createGraphicsPipeline();
  bool createFramebuffers();
  bool createCommandPool();
  bool createCommandBuffers();
  bool createSyncObjects();

  void cleanup();

  // Debug callback
  static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                      VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                      const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                      void* pUserData);
};