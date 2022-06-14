#include "renderlib_wgpu.h"

#include "../renderlib/Logging.h"

#if defined(__APPLE__)
#import <Cocoa/Cocoa.h>
#import <CoreFoundation/CoreFoundation.h>
#import <QuartzCore/CAMetalLayer.h>
#elif defined(__linux__)
#include <X11/Xlib.h>
#include <X11/Xos.h>
#include <X11/Xutil.h>

#endif

#include <string>

static bool renderLibInitialized = false;

static bool renderLibHeadless = false;

static const uint32_t AICS_DEFAULT_STENCIL_BUFFER_BITS = 8;

static const uint32_t AICS_DEFAULT_DEPTH_BUFFER_BITS = 24;

#ifdef __APPLE__
extern "C" static void*
getMetalLayerFromWindow(void* win_id)
{
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
}
#endif

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
renderlib_wgpu::get_surface_id_from_canvas(void* win_id)
{
  // ""
  // "Get an id representing the surface to render to. The way to
  //   obtain this id differs per platform and GUI toolkit.""
  //                                                       "
  WGPUSurfaceDescriptor surface_descriptor;
  surface_descriptor.label = nullptr;

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  WGPUSurfaceDescriptorFromWindowsHWND wgpustruct;
  wgpustruct.hinstance = nullptr;
  wgpustruct.hwnd = win_id;
  wgpustruct.chain.sType = WGPUSType_SurfaceDescriptorFromWindowsHWND;
  surface_descriptor.nextInChain = (const WGPUChainedStruct*)(&wgpustruct);

#elif defined(__APPLE__)
  void* metal_layer_ptr = getMetalLayerFromWindow((void*)win_id);

  WGPUSurfaceDescriptorFromMetalLayer wgpustruct;
  wgpustruct.layer = metal_layer_ptr;
  wgpustruct.chain.sType = WGPUSType_SurfaceDescriptorFromMetalLayer;
  surface_descriptor.nextInChain = (const WGPUChainedStruct*)(&wgpustruct);

#elif __linux__
  Display* display_id = XOpenDisplay(nullptr);
  const char* xdgsessiontype = getenv("XDG_SESSION_TYPE");
  std::string xdgsessiontype_str = xdgsessiontype ? xdgsessiontype : "";
  bool is_wayland = (xdgsessiontype_str.find("wayland") != std::string::npos);
  bool is_xcb = false;
  WGPUSurfaceDescriptorFromWaylandSurface wgpustruct1;
  WGPUSurfaceDescriptorFromXcbWindow wgpustruct2;
  WGPUSurfaceDescriptorFromXlibWindow wgpustruct3;
  if (is_wayland) {
    //# todo: wayland seems to be broken right now
    wgpustruct1.display = display_id;
    wgpustruct1.surface = win_id;
    wgpustruct1.chain.sType = WGPUSType_SurfaceDescriptorFromWaylandSurface;
    surface_descriptor.nextInChain = (const WGPUChainedStruct*)(&wgpustruct1);
  } else if (is_xcb) {
    //# todo: xcb untested
    wgpustruct2.connection = nullptr; // ?? ffi.cast("void *", display_id);
    wgpustruct2.window = *((uint32_t*)(&win_id));
    wgpustruct2.chain.sType = WGPUSType_SurfaceDescriptorFromXlibWindow;
    surface_descriptor.nextInChain = (const WGPUChainedStruct*)(&wgpustruct2);
  } else {
    wgpustruct3.display = display_id;
    wgpustruct3.window = *((uint32_t*)(&win_id));
    wgpustruct3.chain.sType = WGPUSType_SurfaceDescriptorFromXlibWindow;
    surface_descriptor.nextInChain = (const WGPUChainedStruct*)(&wgpustruct3);
  }

#else
  throw("Cannot get surface id: unsupported platform.");
#endif

  WGPUInstance instance_id = nullptr;
  return wgpuInstanceCreateSurface(instance_id, &surface_descriptor);
}
