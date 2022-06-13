#include "renderlib_wgpu.h"

#include "../renderlib/Logging.h"

#ifdef __APPLE__
#import <Cocoa/Cocoa.h>
#import <CoreFoundation/CoreFoundation.h>
#import <QuartzCore/CAMetalLayer.h>

#endif

#include <string>

static bool renderLibInitialized = false;

static bool renderLibHeadless = false;

static const uint32_t AICS_DEFAULT_STENCIL_BUFFER_BITS = 8;

static const uint32_t AICS_DEFAULT_DEPTH_BUFFER_BITS = 24;

static void*
getMetalLayerFromWindow(void* win_id)
{
#ifdef __APPLE__
  // #     id metal_layer = NULL;
  // #     NSWindow *ns_window = glfwGetCocoaWindow(window);
  // #     [ns_window.contentView setWantsLayer:YES];
  // #     metal_layer = [CAMetalLayer layer];
  // #     [ns_window.contentView setLayer:metal_layer];
  // #     surface = wgpu_create_surface_from_metal_layer(metal_layer);
  // # }

  NSView* ns_view = (NSView*)(win_id);
  NSWindow* cw = ns_view.window;
  NSView* cv = cw.contentView;

  CAMetalLayer* metal_layer = nullptr;
  if (cv.layer && [cv.layer isKindOfClass:[CAMetalLayer class]]) {
    //# No need to create a metal layer again
    metal_layer = static_cast<CAMetalLayer*>(cv.layer);
  } else {
    metal_layer = [CAMetalLayer layer];
    cv.layer = metal_layer;
    cv.wantsLayer = true;
  }

  return metal_layer;
#else
  return nullptr;
#endif
}

int
renderlib_wgpu::initialize(bool headless, bool listDevices, int selectedGpu)
{
  if (renderLibInitialized) {
    return 1;
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

  return 0;
}

void
renderlib_wgpu::cleanup()
{
  if (!renderLibInitialized) {
    return;
  }
  LOG_INFO << "Renderlib_wgpu shutdown";

  if (renderLibHeadless) {
  }
  renderLibInitialized = false;
}

WGPUSurface
renderlib_wgpu::get_surface_id_from_canvas(void* win_id, void* display_id)
{
  // ""
  // "Get an id representing the surface to render to. The way to
  //   obtain this id differs per platform and GUI toolkit.""
  //                                                       "

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  WGPUSurfaceDescriptorFromWindowsHWND wgpustruct;
  wgpustruct.hinstance = nullptr;
  wgpustruct.hwnd = win_id;
  wgpustruct.chain.sType = WGPUSType_SurfaceDescriptorFromWindowsHWND;
#elif __APPLE__
  void* metal_layer_ptr = getMetalLayerFromWindow((void*)win_id);

  WGPUSurfaceDescriptorFromMetalLayer wgpustruct;
  wgpustruct.layer = metal_layer_ptr;
  wgpustruct.chain.sType = WGPUSType_SurfaceDescriptorFromMetalLayer;

#elif __linux__
  display_id = this->get_display_id();
  bool is_wayland = "wayland" in os.getenv("XDG_SESSION_TYPE", "").lower();
  bool is_xcb = false;
  if (is_wayland) {
    //# todo: wayland seems to be broken right now
    WGPUSurfaceDescriptorFromWaylandSurface wgpustruct;
    wgpustruct.display = display_id;
    wgpustruct.surface = win_id;
    wgpustruct.chain.sType = WGPUSType_SurfaceDescriptorFromWaylandSurface;
  } else if (is_xcb) {
    //# todo: xcb untested
    WGPUSurfaceDescriptorFromXcbWindow wgpustruct;
    wgpustruct.connection = nullptr; // ?? ffi.cast("void *", display_id);
    wgpustruct.window = win_id;
    wgpustruct.chain.sType = lib.WGPUSType_SurfaceDescriptorFromXlibWindow;
  } else {
    WGPUSurfaceDescriptorFromXlibWindow wgpustruct;
    wgpustruct.display = display_id;
    wgpustruct.window = win_id;
    wgpustruct.chain.sType = WGPUSType_SurfaceDescriptorFromXlibWindow;
  }
#else
  throw("Cannot get surface id: unsupported platform.");
#endif

  WGPUSurfaceDescriptor surface_descriptor;
  surface_descriptor.label = nullptr;
  surface_descriptor.nextInChain = (const WGPUChainedStruct*)(&wgpustruct);

  WGPUInstance instance_id = nullptr;
  return wgpuInstanceCreateSurface(instance_id, &surface_descriptor);
}
