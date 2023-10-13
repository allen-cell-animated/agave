#include "renderlib_wgpu.h"

#include "../renderlib/Logging.h"

#include <string>

static bool renderLibInitialized = false;

static bool renderLibHeadless = false;

static const uint32_t AICS_DEFAULT_STENCIL_BUFFER_BITS = 8;

static const uint32_t AICS_DEFAULT_DEPTH_BUFFER_BITS = 24;

static WGPUInstance sInstance = nullptr;

int
renderlib_wgpu::initialize(bool headless, bool listDevices, int selectedGpu)
{
  if (renderLibInitialized && sInstance) {
    return 1;
  }

  WGPUInstanceDescriptor desc;
  desc.nextInChain = nullptr;
  sInstance = wgpuCreateInstance(&desc);
  if (!sInstance) {
    LOG_ERROR << "Could not initialize WebGPU, wgpuCreateInstance failed!";
    return 0;
  }

  renderLibInitialized = true;

  renderLibHeadless = headless;

  LOG_INFO << "Renderlib_wgpu startup";

  bool enableDebug = false;

  if (headless) {
  } else {
  }

  if (enableDebug) {
  }

  // load gl functions and init stuff

  // then log out some info
  // LOG_INFO << "GL_VENDOR: " << std::string((char*)glGetString(GL_VENDOR));
  // LOG_INFO << "GL_RENDERER: " << std::string((char*)glGetString(GL_RENDERER));

  return 1;
}

WGPUInstance
renderlib_wgpu::getInstance()
{
  return sInstance;
}

void
renderlib_wgpu::cleanup()
{
  if (!renderLibInitialized) {
    return;
  }
  LOG_INFO << "Renderlib_wgpu shutdown";

  wgpuInstanceRelease(sInstance);

  if (renderLibHeadless) {
  }
  renderLibInitialized = false;
}

