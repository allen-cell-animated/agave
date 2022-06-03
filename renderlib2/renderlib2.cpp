#include "renderlib2.h"

#include "Enumerate.hpp"
#include "vk.h"

#include "../renderlib/Logging.h"

#include <iostream>
#include <string>
#include <vector>

static bool renderLibInitialized = false;

static bool renderLibHeadless = false;

static const uint32_t AICS_DEFAULT_STENCIL_BUFFER_BITS = 8;
static const uint32_t AICS_DEFAULT_DEPTH_BUFFER_BITS = 24;

static bool g_validation = true;
static std::vector<const char*> enabledInstanceExtensions;

static VkInstance sInstance = VK_NULL_HANDLE;
static std::vector<VkPhysicalDevice> sPhysicalDevices;

VkInstance
createInstance()
{
  uint32_t requiredExtensionCount = 0;
  // TODO VK_KHR_SURFACE_EXTENSION_NAME
  const char** requiredExtensionNames = nullptr;
  const char* appName = "APP NAME";

  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = appName;
  appInfo.pEngineName = "AGAVE2";
  // TODO: 1.1?  1.2?  check availability vkEnumerateInstanceVersion
  appInfo.apiVersion = VK_API_VERSION_1_0;

  // Vulkan instance creation

  std::vector<const char*> instanceExtensions = { VK_KHR_SURFACE_EXTENSION_NAME };

  // Enable surface extensions depending on os
#if defined(_WIN32)
  instanceExtensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_ANDROID_KHR)
  instanceExtensions.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
#elif defined(_DIRECT2DISPLAY)
  instanceExtensions.push_back(VK_KHR_DISPLAY_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_DIRECTFB_EXT)
  instanceExtensions.push_back(VK_EXT_DIRECTFB_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
  instanceExtensions.push_back(VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_XCB_KHR)
  instanceExtensions.push_back(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_IOS_MVK)
  instanceExtensions.push_back(VK_MVK_IOS_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_MACOS_MVK)
  instanceExtensions.push_back(VK_MVK_MACOS_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_HEADLESS_EXT)
  instanceExtensions.push_back(VK_EXT_HEADLESS_SURFACE_EXTENSION_NAME);
#endif

  enabledInstanceExtensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

  if (enabledInstanceExtensions.size() > 0) {
    for (auto enabledExtension : enabledInstanceExtensions) {
      instanceExtensions.push_back(enabledExtension);
    }
  }

  VkInstanceCreateInfo instanceCreateInfo = {};
  instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instanceCreateInfo.pNext = NULL;
  instanceCreateInfo.pApplicationInfo = &appInfo;
  if (instanceExtensions.size() > 0) {
    if (g_validation) {
      instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    instanceCreateInfo.enabledExtensionCount = (uint32_t)instanceExtensions.size();
    instanceCreateInfo.ppEnabledExtensionNames = instanceExtensions.data();
  }
  std::vector<const char*> validationLayers = {};
  if (g_validation) {
    validationLayers.push_back("VK_LAYER_KHRONOS_validation");
    // The VK_LAYER_KHRONOS_validation contains all current validation functionality.
    // Note that on Android this layer requires at least NDK r20
    // Check if this layer is available at instance level
    uint32_t instanceLayerCount;
    vkEnumerateInstanceLayerProperties(&instanceLayerCount, nullptr);
    std::vector<VkLayerProperties> instanceLayerProperties(instanceLayerCount);
    vkEnumerateInstanceLayerProperties(&instanceLayerCount, instanceLayerProperties.data());
    bool validationLayerPresent = false;

    for (const char* layerName : validationLayers) {
      bool layerFound = false;

      for (const auto& layerProperties : instanceLayerProperties) {
        if (strcmp(layerName, layerProperties.layerName) == 0) {
          layerFound = true;
          break;
        }
      }
      if (!layerFound) {
        LOG_ERROR << "Validation layer " << layerName << " not present, validation is disabled";
        g_validation = false;
        break;
      }
    }
  }
  if (g_validation) {
    instanceCreateInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
    instanceCreateInfo.ppEnabledLayerNames = validationLayers.data();
  } else {
    instanceCreateInfo.enabledLayerCount = 0;
  }

  VkInstance instance = nullptr;
  VK_CHECK(vkCreateInstance(&instanceCreateInfo, nullptr, &instance));

  return instance;
}

void
GetVulkanPhysicalDevices()
{
  GetEnumerateVector(sInstance, vkEnumeratePhysicalDevices, sPhysicalDevices);

  for (auto device : sPhysicalDevices) {
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(device, &deviceProperties);
    LOG_INFO << "Physical device: " << deviceProperties.deviceName;
  }
}

int
renderlib2::initialize(bool headless, bool listDevices, int selectedGpu)
{
  if (renderLibInitialized) {
    return 1;
  }

  LOG_INFO << "Renderlib2 startup";

  sInstance = createInstance();
  if (!sInstance) {
    LOG_ERROR << "Failed to create Vulkan instance";
  }

  GetVulkanPhysicalDevices();
  if (sPhysicalDevices.empty()) {
    LOG_ERROR << "Found no Vulkan physical devices";
  }

  renderLibInitialized = true;

  renderLibHeadless = headless;

  bool enableDebug = false;

  if (headless) {
  } else {
  }

  if (enableDebug) {
  }

  int status = (sInstance != VK_NULL_HANDLE) ? 1 : 0;
  return status;
}

VkInstance
renderlib2::instance()
{
  return sInstance;
}

void
renderlib2::clearGpuVolumeCache()
{
}

void
renderlib2::cleanup()
{
  if (!renderLibInitialized) {
    return;
  }
  LOG_INFO << "Renderlib2 shutdown";

  clearGpuVolumeCache();

  renderLibInitialized = false;
}
