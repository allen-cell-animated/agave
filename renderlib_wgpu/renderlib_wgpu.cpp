#include "renderlib_wgpu.h"

#include "../renderlib/Logging.h"

#include "getsurface_wgpu.h"
#include "wgpu_util.h"

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

WGPUSurface
renderlib_wgpu::getSurfaceFromCanvas(void* win_id)
{
  WGPUSurface surface = get_surface_from_canvas(getInstance(), win_id);
  return surface;
}

WGPUAdapter
renderlib_wgpu::getAdapter(WGPUSurface surface)
{
  WGPUAdapter adapter;
  WGPURequestAdapterOptions options = {
    .nextInChain = NULL,
    .compatibleSurface = surface,
  };

  wgpuInstanceRequestAdapter(getInstance(), &options, request_adapter_callback, (void*)&adapter);

  printAdapterFeatures(adapter);

  return adapter;
}

WGPUDevice
renderlib_wgpu::requestDevice(WGPUAdapter adapter)
{
  WGPUDevice device;
  WGPURequiredLimits requiredLimits = {
    .nextInChain = NULL,
    .limits =
      WGPULimits{
        .maxBindGroups = 1,
      },
  };
  WGPUDeviceExtras deviceExtras = {
    .chain =
      WGPUChainedStruct{
        .next = NULL,
        .sType = (WGPUSType)WGPUSType_DeviceExtras,
      },
    .tracePath = NULL,
  };
  WGPUUncapturedErrorCallbackInfo uncapturedErrorInfo = {
    .nextInChain = nullptr,
    .callback = handle_uncaptured_error,
    .userdata = NULL,
  };

  WGPUDeviceDescriptor deviceDescriptor = { .nextInChain = (const WGPUChainedStruct*)&deviceExtras,
                                            .label = "AGAVE wgpu device",
                                            .requiredFeatureCount = 0,
                                            .requiredLimits = nullptr, // & requiredLimits,
                                            .defaultQueue =
                                              WGPUQueueDescriptor{
                                                .nextInChain = NULL,
                                                .label = "AGAVE default wgpu queue",
                                              },
                                            .deviceLostCallback = handle_device_lost,
                                            .deviceLostUserdata = NULL,
                                            .uncapturedErrorCallbackInfo = uncapturedErrorInfo };

  // creates/ fills in m_device!
  wgpuAdapterRequestDevice(adapter, &deviceDescriptor, request_device_callback, (void*)&device);

  return device;
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
