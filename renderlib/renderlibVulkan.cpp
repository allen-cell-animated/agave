#include "renderlibVulkan.h"

#include "ImageXYZC.h"
#include "ImageGpuVK.h"
#include "graphicsVulkan/ImageXyzcGpuVK.h"
#include "graphicsVulkan/RenderVK.h"
#include "graphicsVulkan/RenderVKPT.h"
#include "Logging.h"

#include <string>
#include <set>
#include <algorithm>
#include <cstring>

#if defined(_WIN32) || defined(_WIN64)
#ifdef __cplusplus
extern "C"
{
#endif

  __declspec(dllexport) DWORD NvOptimusEnablement = 1;
  __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;

#ifdef __cplusplus
}
#endif
#endif

// Static member definitions
static bool renderLibVKInitialized = false;
static bool renderLibVKHeadless = false;
static std::string s_assetPath = "";

std::map<std::shared_ptr<ImageXYZC>, std::shared_ptr<ImageGpuVK>> renderlibVK::sGpuImageCache;

VkInstance renderlibVK::s_vulkanInstance = nullptr;
VkDevice renderlibVK::s_device = VK_NULL_HANDLE;
VkPhysicalDevice renderlibVK::s_physicalDevice = VK_NULL_HANDLE;
VkQueue renderlibVK::s_graphicsQueue = VK_NULL_HANDLE;
VkQueue renderlibVK::s_computeQueue = VK_NULL_HANDLE;
VkCommandPool renderlibVK::s_commandPool = VK_NULL_HANDLE;
uint32_t renderlibVK::s_graphicsQueueFamily = UINT32_MAX;
uint32_t renderlibVK::s_computeQueueFamily = UINT32_MAX;
VkDebugUtilsMessengerEXT renderlibVK::s_debugMessenger = VK_NULL_HANDLE;

// Validation layers
const std::vector<const char*> validationLayers = { "VK_LAYER_KHRONOS_validation" };

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

int
renderlibVK::initialize(std::string assetPath, bool headless, bool listDevices, int selectedGpu)
{
  if (renderLibVKInitialized) {
    return 1;
  }
  renderLibVKInitialized = true;
  s_assetPath = assetPath;
  renderLibVKHeadless = headless;

  LOG_INFO << "RenderlibVK startup";

  // Create Vulkan instance
  if (!createVulkanInstance(enableValidationLayers)) {
    LOG_ERROR << "Failed to create Vulkan instance";
    return 0;
  }

  // List devices if requested
  if (listDevices) {
    listVulkanDevices();
    return 1;
  }

  // Select physical device
  if (!selectPhysicalDevice(selectedGpu)) {
    LOG_ERROR << "Failed to find suitable GPU";
    return 0;
  }

  // Create logical device
  if (!createLogicalDevice()) {
    LOG_ERROR << "Failed to create logical device";
    return 0;
  }

  // Create command pool
  if (!createCommandPool()) {
    LOG_ERROR << "Failed to create command pool";
    return 0;
  }

  LOG_INFO << "Vulkan initialization complete";
  return 1;
}

std::string
renderlibVK::assetPath()
{
  return s_assetPath;
}

void
renderlibVK::clearGpuVolumeCache()
{
  // Clean up the shared GPU buffer cache
  for (auto& i : sGpuImageCache) {
    i.second->deallocGpu();
  }
  sGpuImageCache.clear();
}

void
renderlibVK::cleanup()
{
  if (!renderLibVKInitialized) {
    return;
  }
  LOG_INFO << "RenderlibVK shutdown";

  clearGpuVolumeCache();

  // Cleanup Vulkan objects
  if (s_device != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(s_device);

    if (s_commandPool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(s_device, s_commandPool, nullptr);
      s_commandPool = VK_NULL_HANDLE;
    }

    vkDestroyDevice(s_device, nullptr);
    s_device = VK_NULL_HANDLE;
  }

  // Cleanup debug messenger
  if (s_debugMessenger != VK_NULL_HANDLE && s_vulkanInstance) {
    vkDestroyDebugUtilsMessengerEXT(s_vulkanInstance, s_debugMessenger, nullptr);
    s_debugMessenger = VK_NULL_HANDLE;
  }

  vkDestroyInstance(s_vulkanInstance, nullptr);
  s_vulkanInstance = nullptr;

  renderLibVKInitialized = false;
}

std::shared_ptr<ImageGpuVK>
renderlibVK::imageAllocGPU(std::shared_ptr<ImageXYZC> image, bool do_cache)
{
  auto cached = sGpuImageCache.find(image);
  if (cached != sGpuImageCache.end()) {
    return cached->second;
  }

  ImageXyzcGpuVK* cimg = new ImageXyzcGpuVK;
  cimg->allocGpuInterleaved(s_device, s_physicalDevice, s_commandPool, s_graphicsQueue, image.get());
  std::shared_ptr<ImageGpuVK> shared(cimg);

  if (do_cache) {
    sGpuImageCache[image] = shared;
  }

  return shared;
}

void
renderlibVK::imageDeallocGPU(std::shared_ptr<ImageXYZC> image)
{
  auto cached = sGpuImageCache.find(image);
  if (cached != sGpuImageCache.end()) {
    cached->second->deallocGpu();
    sGpuImageCache.erase(image);
  }
}

VkInstance
renderlibVK::getVulkanInstance()
{
  return s_vulkanInstance;
}
VkDevice
renderlibVK::getVulkanDevice()
{
  return s_device;
}
VkPhysicalDevice
renderlibVK::getVulkanPhysicalDevice()
{
  return s_physicalDevice;
}
VkQueue
renderlibVK::getGraphicsQueue()
{
  return s_graphicsQueue;
}
VkQueue
renderlibVK::getComputeQueue()
{
  return s_computeQueue;
}
VkCommandPool
renderlibVK::getCommandPool()
{
  return s_commandPool;
}

bool
renderlibVK::isVulkanSupported()
{
  return s_vulkanInstance != nullptr && s_device != VK_NULL_HANDLE;
}

std::vector<VkPhysicalDevice>
renderlibVK::getAvailableGPUs()
{
  std::vector<VkPhysicalDevice> devices;

  if (!s_vulkanInstance) {
    return devices;
  }

  uint32_t deviceCount = 0;
  vkEnumeratePhysicalDevices(s_vulkanInstance, &deviceCount, nullptr);

  if (deviceCount > 0) {
    devices.resize(deviceCount);
    vkEnumeratePhysicalDevices(s_vulkanInstance, &deviceCount, devices.data());
  }

  return devices;
}

void
renderlibVK::listVulkanDevices()
{
  auto devices = getAvailableGPUs();

  LOG_INFO << devices.size() << " Vulkan devices found:";

  for (size_t i = 0; i < devices.size(); i++) {
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(devices[i], &deviceProperties);

    LOG_INFO << "Device " << i << ": " << deviceProperties.deviceName;
    LOG_INFO << "  Type: " << deviceProperties.deviceType;
    LOG_INFO << "  API Version: " << VK_VERSION_MAJOR(deviceProperties.apiVersion) << "."
             << VK_VERSION_MINOR(deviceProperties.apiVersion) << "." << VK_VERSION_PATCH(deviceProperties.apiVersion);
    LOG_INFO << "  Driver Version: " << deviceProperties.driverVersion;
    LOG_INFO << "  Vendor ID: " << std::hex << deviceProperties.vendorID;
    LOG_INFO << "  Device ID: " << std::hex << deviceProperties.deviceID;
  }
}

bool
renderlibVK::createVulkanInstance(bool enableValidation)
{

  VkInstanceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "RenderlibVK";
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.pEngineName = "RenderlibVKEngine";
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.apiVersion = VK_API_VERSION_1_4;
  createInfo.pApplicationInfo = &appInfo;

  VkAllocationCallbacks* allocator = nullptr;

  if (enableValidation && checkValidationLayerSupport()) {
    createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();
  }

  // Set required extensions
  auto extensions = getRequiredExtensions();
  createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  createInfo.ppEnabledExtensionNames = extensions.data();

  VkResult result = vkCreateInstance(&createInfo, allocator, &s_vulkanInstance);
  if (result != VK_SUCCESS) {
    LOG_ERROR << "Failed to create Vulkan instance";
    s_vulkanInstance = nullptr;
    return false;
  }

  LOG_INFO << "Created Vulkan instance";
  return true;
}

bool
renderlibVK::selectPhysicalDevice(int selectedGpu)
{
  auto devices = getAvailableGPUs();

  if (devices.empty()) {
    LOG_ERROR << "No Vulkan devices available";
    return false;
  }

  if (selectedGpu >= 0 && selectedGpu < static_cast<int>(devices.size())) {
    s_physicalDevice = devices[selectedGpu];
  } else {
    // Select first discrete GPU, or first GPU if no discrete found
    for (const auto& device : devices) {
      VkPhysicalDeviceProperties deviceProperties;
      vkGetPhysicalDeviceProperties(device, &deviceProperties);

      if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        s_physicalDevice = device;
        break;
      }
    }

    if (s_physicalDevice == VK_NULL_HANDLE) {
      s_physicalDevice = devices[0];
    }
  }

  VkPhysicalDeviceProperties deviceProperties;
  vkGetPhysicalDeviceProperties(s_physicalDevice, &deviceProperties);
  LOG_INFO << "Selected GPU: " << deviceProperties.deviceName;

  return true;
}

bool
renderlibVK::createLogicalDevice()
{
  // Find queue families
  s_graphicsQueueFamily = findQueueFamilies(s_physicalDevice, VK_QUEUE_GRAPHICS_BIT);
  s_computeQueueFamily = findQueueFamilies(s_physicalDevice, VK_QUEUE_COMPUTE_BIT);

  if (s_graphicsQueueFamily == UINT32_MAX) {
    LOG_ERROR << "No graphics queue family found";
    return false;
  }

  std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
  std::set<uint32_t> uniqueQueueFamilies = { s_graphicsQueueFamily };

  if (s_computeQueueFamily != UINT32_MAX) {
    uniqueQueueFamilies.insert(s_computeQueueFamily);
  } else {
    s_computeQueueFamily = s_graphicsQueueFamily; // Fallback to graphics queue
  }

  float queuePriority = 1.0f;
  for (uint32_t queueFamily : uniqueQueueFamilies) {
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = queueFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    queueCreateInfos.push_back(queueCreateInfo);
  }

  VkPhysicalDeviceFeatures deviceFeatures{};
  deviceFeatures.samplerAnisotropy = VK_TRUE;
  deviceFeatures.shaderStorageImageExtendedFormats = VK_TRUE;

  VkDeviceCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
  createInfo.pQueueCreateInfos = queueCreateInfos.data();
  createInfo.pEnabledFeatures = &deviceFeatures;

  // Enable extensions
  std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
  createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
  createInfo.ppEnabledExtensionNames = deviceExtensions.data();

  // Enable validation layers for device (compatibility)
  if (enableValidationLayers) {
    createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
    createInfo.ppEnabledLayerNames = validationLayers.data();
  } else {
    createInfo.enabledLayerCount = 0;
  }

  VkResult result = vkCreateDevice(s_physicalDevice, &createInfo, nullptr, &s_device);
  if (result != VK_SUCCESS) {
    LOG_ERROR << "Failed to create logical device: " << result;
    return false;
  }

  // Get queue handles
  vkGetDeviceQueue(s_device, s_graphicsQueueFamily, 0, &s_graphicsQueue);
  vkGetDeviceQueue(s_device, s_computeQueueFamily, 0, &s_computeQueue);

  LOG_INFO << "Created Vulkan logical device";
  return true;
}

bool
renderlibVK::createCommandPool()
{
  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = s_graphicsQueueFamily;

  VkResult result = vkCreateCommandPool(s_device, &poolInfo, nullptr, &s_commandPool);
  if (result != VK_SUCCESS) {
    LOG_ERROR << "Failed to create command pool: " << result;
    return false;
  }

  return true;
}

uint32_t
renderlibVK::findQueueFamilies(VkPhysicalDevice device, VkQueueFlags queueFlags)
{
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

  for (uint32_t i = 0; i < queueFamilyCount; i++) {
    if (queueFamilies[i].queueFlags & queueFlags) {
      return i;
    }
  }

  return UINT32_MAX;
}

bool
renderlibVK::checkValidationLayerSupport()
{
  uint32_t layerCount;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

  std::vector<VkLayerProperties> availableLayers(layerCount);
  vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

  for (const char* layerName : validationLayers) {
    bool layerFound = false;

    for (const auto& layerProperties : availableLayers) {
      if (strcmp(layerName, layerProperties.layerName) == 0) {
        layerFound = true;
        break;
      }
    }

    if (!layerFound) {
      return false;
    }
  }

  return true;
}

std::vector<const char*>
renderlibVK::getRequiredExtensions()
{
  std::vector<const char*> extensions;

  if (enableValidationLayers) {
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
  }

  return extensions;
}

VKAPI_ATTR VkBool32 VKAPI_CALL
renderlibVK::debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                           VkDebugUtilsMessageTypeFlagsEXT messageType,
                           const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                           void* pUserData)
{
  if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
    LOG_WARNING << "Vulkan validation: " << pCallbackData->pMessage;
  }

  return VK_FALSE;
}

IVulkanRenderWindow*
renderlibVK::createRenderer(renderlibVK::RendererType rendererType, RenderSettings* rs)
{
  switch (rendererType) {
    case renderlibVK::RendererType::RendererType_Raymarch: {
      RenderVK* renderer = new RenderVK(rs);
      return renderer;
    }
    case renderlibVK::RendererType::RendererType_Pathtrace:
    default: {
      RenderVKPT* renderer = new RenderVKPT(rs);
      return renderer;
    }
  }
}

renderlibVK::RendererType
renderlibVK::stringToRendererType(std::string rendererTypeString)
{
  if (rendererTypeString == "RenderVK") {
    return RendererType::RendererType_Raymarch;
  } else if (rendererTypeString == "RenderVKPT") {
    return RendererType::RendererType_Pathtrace;
  } else {
    return RendererType::RendererType_Pathtrace;
  }
}

std::string
renderlibVK::rendererTypeToString(renderlibVK::RendererType rendererType)
{
  switch (rendererType) {
    case renderlibVK::RendererType_Raymarch:
      return "RenderVK";
    case renderlibVK::RendererType_Pathtrace:
    default:
      return "RenderVKPT";
  }
}

// HeadlessVKContext implementation
HeadlessVKContext::HeadlessVKContext()
  : m_device(VK_NULL_HANDLE)
  , m_queue(VK_NULL_HANDLE)
  , m_commandPool(VK_NULL_HANDLE)
  , m_queueFamilyIndex(UINT32_MAX)
{
}

HeadlessVKContext::~HeadlessVKContext()
{
  cleanup();
}

bool
HeadlessVKContext::initialize(VkPhysicalDevice physicalDevice)
{
  // Implementation would create a minimal Vulkan device for headless operation
  LOG_INFO << "Initializing headless Vulkan context";
  return true;
}

void
HeadlessVKContext::cleanup()
{
  if (m_device != VK_NULL_HANDLE) {
    if (m_commandPool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(m_device, m_commandPool, nullptr);
      m_commandPool = VK_NULL_HANDLE;
    }

    vkDestroyDevice(m_device, nullptr);
    m_device = VK_NULL_HANDLE;
  }
}

// RendererVKContext implementation
RendererVKContext::RendererVKContext()
  : m_ownDevice(true)
  , m_device(VK_NULL_HANDLE)
  , m_graphicsQueue(VK_NULL_HANDLE)
  , m_computeQueue(VK_NULL_HANDLE)
  , m_commandPool(VK_NULL_HANDLE)
  , m_headlessContext(nullptr)
{
}

RendererVKContext::~RendererVKContext()
{
  destroy();
}

void
RendererVKContext::configure(VkDevice device)
{
  if (device != VK_NULL_HANDLE) {
    m_device = device;
    m_ownDevice = false;
  }
}

void
RendererVKContext::init()
{
  if (renderLibVKHeadless) {
    m_headlessContext = new HeadlessVKContext();
    m_headlessContext->initialize(renderlibVK::getVulkanPhysicalDevice());
    m_device = m_headlessContext->getDevice();
  } else {
    initVulkanDevice();
  }
}

void
RendererVKContext::destroy()
{
  if (m_headlessContext) {
    delete m_headlessContext;
    m_headlessContext = nullptr;
  }

  if (m_ownDevice && m_device != VK_NULL_HANDLE) {
    if (m_commandPool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(m_device, m_commandPool, nullptr);
      m_commandPool = VK_NULL_HANDLE;
    }

    vkDestroyDevice(m_device, nullptr);
    m_device = VK_NULL_HANDLE;
  }
}

VkDevice
RendererVKContext::getDevice() const
{
  return m_device;
}
VkQueue
RendererVKContext::getGraphicsQueue() const
{
  return m_graphicsQueue;
}
VkQueue
RendererVKContext::getComputeQueue() const
{
  return m_computeQueue;
}
VkCommandPool
RendererVKContext::getCommandPool() const
{
  return m_commandPool;
}

VkCommandBuffer
RendererVKContext::beginSingleTimeCommands()
{
  VkCommandBufferAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandPool = m_commandPool;
  allocInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer;
  vkAllocateCommandBuffers(m_device, &allocInfo, &commandBuffer);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(commandBuffer, &beginInfo);
  return commandBuffer;
}

void
RendererVKContext::endSingleTimeCommands(VkCommandBuffer commandBuffer)
{
  vkEndCommandBuffer(commandBuffer);

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
  vkQueueWaitIdle(m_graphicsQueue);

  vkFreeCommandBuffers(m_device, m_commandPool, 1, &commandBuffer);
}

void
RendererVKContext::initVulkanDevice()
{
  // Use shared Vulkan resources from renderlibVK
  m_device = renderlibVK::getVulkanDevice();
  m_graphicsQueue = renderlibVK::getGraphicsQueue();
  m_computeQueue = renderlibVK::getComputeQueue();
  m_commandPool = renderlibVK::getCommandPool();
  m_ownDevice = false;
}