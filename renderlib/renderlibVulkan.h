#pragma once

#include <vulkan/vulkan.h>

#include <QVulkanInstance>
#include <QWindow>

#include <map>
#include <memory>
#include <string>
#include <vector>

struct ImageGpuVK;
class ImageXYZC;
class IVulkanRenderWindow;
class RenderSettings;

class renderlibVK
{
public:
  static int initialize(std::string assetPath, bool headless = false, bool listDevices = false, int selectedGpu = 0);
  static void clearGpuVolumeCache();
  static void cleanup();

  static std::string assetPath();

  // Vulkan GPU cache management
  // Similar to OpenGL version but uses Vulkan resources
  static std::shared_ptr<ImageGpuVK> imageAllocGPU(std::shared_ptr<ImageXYZC> image, bool do_cache = true);
  static void imageDeallocGPU(std::shared_ptr<ImageXYZC> image);

  static QVulkanInstance* getVulkanInstance();
  static VkDevice getVulkanDevice();
  static VkPhysicalDevice getVulkanPhysicalDevice();
  static VkQueue getGraphicsQueue();
  static VkQueue getComputeQueue();
  static VkCommandPool getCommandPool();

  enum RendererType
  {
    RendererType_Pathtrace,
    RendererType_Raymarch
  };

  // Factory method for creating Vulkan renderers
  static IVulkanRenderWindow* createRenderer(RendererType rendererType, RenderSettings* rs = nullptr);
  static RendererType stringToRendererType(std::string rendererTypeString);
  static std::string rendererTypeToString(RendererType rendererType);

  // Vulkan-specific utilities
  static bool isVulkanSupported();
  static std::vector<VkPhysicalDevice> getAvailableGPUs();
  static void listVulkanDevices();
  static bool selectGPU(int selectedGpu);

private:
  static std::map<std::shared_ptr<ImageXYZC>, std::shared_ptr<ImageGpuVK>> sGpuImageCache;

  // Vulkan state
  static QVulkanInstance* s_vulkanInstance;
  static VkDevice s_device;
  static VkPhysicalDevice s_physicalDevice;
  static VkQueue s_graphicsQueue;
  static VkQueue s_computeQueue;
  static VkCommandPool s_commandPool;
  static uint32_t s_graphicsQueueFamily;
  static uint32_t s_computeQueueFamily;

  // Helper methods
  static bool createVulkanInstance(bool enableValidation = false);
  static bool selectPhysicalDevice(int selectedGpu);
  static bool createLogicalDevice();
  static bool createCommandPool();
  static uint32_t findQueueFamilies(VkPhysicalDevice device, VkQueueFlags queueFlags);
  static bool checkValidationLayerSupport();
  static std::vector<const char*> getRequiredExtensions();
  static void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);

  // Debug callback
  static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                      VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                      const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                      void* pUserData);

  static VkDebugUtilsMessengerEXT s_debugMessenger;
};

// Vulkan headless context for offscreen rendering
class HeadlessVKContext
{
public:
  HeadlessVKContext();
  ~HeadlessVKContext();

  bool initialize(VkPhysicalDevice physicalDevice);
  void cleanup();

  VkDevice getDevice() const { return m_device; }
  VkQueue getQueue() const { return m_queue; }
  VkCommandPool getCommandPool() const { return m_commandPool; }

private:
  VkDevice m_device;
  VkQueue m_queue;
  VkCommandPool m_commandPool;
  uint32_t m_queueFamilyIndex;
};

// Vulkan context wrapper for thread management
class RendererVKContext
{
public:
  RendererVKContext();
  ~RendererVKContext();

  void configure(VkDevice device = VK_NULL_HANDLE);
  void init();
  void destroy();

  VkDevice getDevice() const;
  VkQueue getGraphicsQueue() const;
  VkQueue getComputeQueue() const;
  VkCommandPool getCommandPool() const;

  // Command buffer management for threading
  VkCommandBuffer beginSingleTimeCommands();
  void endSingleTimeCommands(VkCommandBuffer commandBuffer);

private:
  bool m_ownDevice;
  VkDevice m_device;
  VkQueue m_graphicsQueue;
  VkQueue m_computeQueue;
  VkCommandPool m_commandPool;

  HeadlessVKContext* m_headlessContext;

  void initVulkanDevice();
};