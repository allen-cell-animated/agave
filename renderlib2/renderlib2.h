#pragma once

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
