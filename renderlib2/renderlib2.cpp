#include "renderlib2.h"

#include "../renderlib/Logging.h"

#include <iostream>
#include <string>
#include <vector>

// we want to immediately abort when there is an error.
// In normal engines this would give an error message to the user, or perform a dump of state.
#define VK_CHECK(x)                                                                                                    \
  do {                                                                                                                 \
    VkResult err = x;                                                                                                  \
    if (err) {                                                                                                         \
      LOG_ERROR << "Detected Vulkan error: " << err;                                                                   \
      abort();                                                                                                         \
    }                                                                                                                  \
  } while (0)

static bool renderLibInitialized = false;

static bool renderLibHeadless = false;

static const uint32_t AICS_DEFAULT_STENCIL_BUFFER_BITS = 8;
static const uint32_t AICS_DEFAULT_DEPTH_BUFFER_BITS = 24;

static bool g_validation = true;
static std::vector<const char*> enabledInstanceExtensions;
static VkInstance sInstance = VK_NULL_HANDLE;

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

  /***
    // Enable surface extensions depending on os
  #if defined(_WIN32)
    instanceExtensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
  #elif defined(VK_USE_PLATFORM_ANDROID_KHR)
    instanceExtensions.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
  #elif defined(_DIRECT2DISPLAY)
    instanceExtensions.push_back(VK_KHR_DISPLAY_EXTENSION_NAME);
  #elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
    instanceExtensions.push_back(VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME);
  #elif defined(VK_USE_PLATFORM_XCB_KHR)
    instanceExtensions.push_back(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
  #elif defined(VK_USE_PLATFORM_IOS_MVK)
    instanceExtensions.push_back(VK_MVK_IOS_SURFACE_EXTENSION_NAME);
  #elif defined(VK_USE_PLATFORM_MACOS_MVK)
    instanceExtensions.push_back(VK_MVK_MACOS_SURFACE_EXTENSION_NAME);
  #endif
  ****/

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
  if (g_validation) {
    // The VK_LAYER_KHRONOS_validation contains all current validation functionality.
    // Note that on Android this layer requires at least NDK r20
    const char* validationLayerName = "VK_LAYER_KHRONOS_validation";
    // Check if this layer is available at instance level
    uint32_t instanceLayerCount;
    vkEnumerateInstanceLayerProperties(&instanceLayerCount, nullptr);
    std::vector<VkLayerProperties> instanceLayerProperties(instanceLayerCount);
    vkEnumerateInstanceLayerProperties(&instanceLayerCount, instanceLayerProperties.data());
    bool validationLayerPresent = false;
    for (VkLayerProperties layer : instanceLayerProperties) {
      if (strcmp(layer.layerName, validationLayerName) == 0) {
        validationLayerPresent = true;
        break;
      }
    }
    if (validationLayerPresent) {
      instanceCreateInfo.ppEnabledLayerNames = &validationLayerName;
      instanceCreateInfo.enabledLayerCount = 1;
    } else {
      std::cerr << "Validation layer VK_LAYER_KHRONOS_validation not present, validation is disabled";
    }
  }

  VkInstance instance = nullptr;
  VK_CHECK(vkCreateInstance(&instanceCreateInfo, nullptr, &instance));

  return instance;
}

int
renderlib2::initialize(bool headless, bool listDevices, int selectedGpu)
{
  if (renderLibInitialized) {
    return 1;
  }

  sInstance = createInstance();

  renderLibInitialized = true;

  renderLibHeadless = headless;

  LOG_INFO << "Renderlib2 startup";

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
{}

void
renderlib2::cleanup()
{
  if (!renderLibInitialized) {
    return;
  }
  LOG_INFO << "Renderlib shutdown";

  clearGpuVolumeCache();

  renderLibInitialized = false;
}
