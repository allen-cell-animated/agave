#pragma once

#include "webgpu-headers/webgpu.h"
#include "wgpu.h"

#include <map>
#include <memory>
#include <string>

class WgpuWindowContext
{

public:
  WgpuWindowContext();
  ~WgpuWindowContext();
  WGPUSurface m_surface;
  WGPUSurfaceConfiguration m_surfaceConfig;
  WGPUTextureFormat m_surfaceFormat;

  void resize(uint32_t width, uint32_t height);
  void present();
};

class renderlib_wgpu
{
public:
  static int initialize(bool headless = false, bool listDevices = false, int selectedGpu = 0);
  static void cleanup();

  static WGPUInstance getInstance();
  static WGPUSurface getSurfaceFromCanvas(void* win_id);
  static WGPUAdapter getAdapter(WGPUSurface surface);
  static WGPUDevice requestDevice(WGPUAdapter adapter);
  static WgpuWindowContext* setupWindowContext(WGPUSurface surface, WGPUDevice device, uint32_t width, uint32_t height);
};
