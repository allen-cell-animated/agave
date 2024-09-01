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

  WGPUInstanceDescriptor desc = {};
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
  WGPURequestAdapterOptions options = {};
  options.nextInChain = NULL;
  options.compatibleSurface = surface;

  wgpuInstanceRequestAdapter(getInstance(), &options, request_adapter_callback, (void*)&adapter);

  printAdapterFeatures(adapter);

  return adapter;
}

WGPUDevice
renderlib_wgpu::requestDevice(WGPUAdapter adapter)
{
  WGPUDevice device;

  WGPULimits limits;
  limits.maxBindGroups = 1;

  WGPURequiredLimits requiredLimits = {};
  requiredLimits.nextInChain = NULL;
  requiredLimits.limits = limits;

  WGPUChainedStruct chain = {};
  chain.next = NULL;
  chain.sType = (WGPUSType)WGPUSType_DeviceExtras;
  WGPUDeviceExtras deviceExtras = {};
  deviceExtras.chain = chain;
  deviceExtras.tracePath = NULL;

  WGPUUncapturedErrorCallbackInfo uncapturedErrorInfo = {};
  uncapturedErrorInfo.nextInChain = nullptr;
  uncapturedErrorInfo.callback = handle_uncaptured_error;
  uncapturedErrorInfo.userdata = NULL;

  WGPUQueueDescriptor queueDescriptor = {};
  queueDescriptor.nextInChain = NULL;
  queueDescriptor.label = "AGAVE default wgpu queue";

  WGPUDeviceDescriptor deviceDescriptor = {};
  deviceDescriptor.nextInChain = (const WGPUChainedStruct*)&deviceExtras;
  deviceDescriptor.label = "AGAVE wgpu device";
  deviceDescriptor.requiredFeatureCount = 0;
  deviceDescriptor.requiredLimits = nullptr;
  deviceDescriptor.defaultQueue = queueDescriptor;
  deviceDescriptor.deviceLostCallback = handle_device_lost;
  deviceDescriptor.deviceLostUserdata = NULL;
  deviceDescriptor.uncapturedErrorCallbackInfo = uncapturedErrorInfo;

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
