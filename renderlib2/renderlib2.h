#pragma once

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#define VK_USE_PLATFORM_WIN32_KHR
#elif __APPLE__
#define VK_USE_PLATFORM_MACOS_MVK
#elif __linux__
#define VK_USE_PLATFORM_XCB_KHR
#else
#error "Unknown platform to set Vulkan platform defines"
#endif

#include <vulkan/vulkan.h>

#include <map>
#include <memory>
#include <string>

class renderlib2
{
public:
  static int initialize(bool headless = false, bool listDevices = false, int selectedGpu = 0);
  static VkInstance instance();
  static void clearGpuVolumeCache();
  static void cleanup();

private:
};
