#pragma once

#include "webgpu-headers/webgpu.h"
#include "wgpu.h"

#ifdef __cplusplus
extern "C"
{
#endif

  /**
   * Get a WGPUSurface from a window handle.
   */
  WGPUSurface get_surface_from_canvas(WGPUInstance instance, void* win_id);

#ifdef __cplusplus
}
#endif
