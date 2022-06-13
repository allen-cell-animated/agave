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

  static WGPUSurface get_surface_id_from_canvas(void* win_id, void* display_id);

  static void* getMetalLayerFromWindow(void* win_id);
};
