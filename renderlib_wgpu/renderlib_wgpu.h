#pragma once

#include "webgpu-headers/webgpu.h"
#include "wgpu.h"

#include <map>
#include <memory>
#include <string>

class renderlib_wgpu
{
public:
  static int initialize(bool headless = false, bool listDevices = false, int selectedGpu = 0);
  static void cleanup();

  static WGPUInstance getInstance();
};