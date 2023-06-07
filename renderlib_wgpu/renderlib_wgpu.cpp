#include "renderlib_wgpu.h"

#include "../renderlib/Logging.h"

#if defined(__APPLE__)
#import <Cocoa/Cocoa.h>
#import <Foundation/Foundation.h>
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

static WGPUInstance sInstance = nullptr;

#ifdef __APPLE__
static void*
getMetalLayerFromWindow(void* win_id)
{
  // #     id metal_layer = NULL;
  // #     NSWindow *ns_window = glfwGetCocoaWindow(window);
  // #     [ns_window.contentView setWantsLayer:YES];
  // #     metal_layer = [CAMetalLayer layer];
  // #     [ns_window.contentView setLayer:metal_layer];
  // #     surface = wgpu_create_surface_from_metal_layer(metal_layer);
  // # }

  NSView* view = (NSView*)(win_id);
  if (![view isKindOfClass:[NSView class]]) {
    LOG_ERROR << "Failed to get NSView from Qt window id";
  }

  if (![view.layer isKindOfClass:[CAMetalLayer class]]) {
    // orilayer = [view layer];
    [view setLayer:[CAMetalLayer layer]];
    [view setWantsLayer:YES];
  }

  // TODO cleanup later?
  return [view layer];
}
#endif

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

  wgpuInstanceDrop(sInstance);

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
  surface_descriptor.label = "AGAVE wgpu surface";

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
  HINSTANCE hinstance = GetModuleHandle(NULL);
  WGPUSurfaceDescriptorFromWindowsHWND wgpustruct;
  wgpustruct.hinstance = hinstance;
  wgpustruct.hwnd = win_id;
  wgpustruct.chain.next = nullptr;
  wgpustruct.chain.sType = WGPUSType_SurfaceDescriptorFromWindowsHWND;
  surface_descriptor.nextInChain = (const WGPUChainedStruct*)(&wgpustruct);

#elif defined(__APPLE__)
  void* metal_layer_ptr = getMetalLayerFromWindow((void*)win_id);

  WGPUSurfaceDescriptorFromMetalLayer wgpustruct;
  wgpustruct.layer = metal_layer_ptr;
  wgpustruct.chain.next = nullptr;
  wgpustruct.chain.sType = WGPUSType_SurfaceDescriptorFromMetalLayer;
  surface_descriptor.nextInChain = (const WGPUChainedStruct*)(&wgpustruct);

#elif __linux__
  Display* display_id = XOpenDisplay(nullptr);
  const char* xdgsessiontype = getenv("XDG_SESSION_TYPE");
  std::string xdgsessiontype_str = xdgsessiontype ? xdgsessiontype : "";
  bool is_wayland = false;
  bool is_x11 = false;
  bool is_xcb = false;
  if (xdgsessiontype_str.empty()) {
    const char* waylanddisplayenv = getenv("WAYLAND_DISPLAY");
    std::string waylanddisplay_str = waylanddisplayenv ? waylanddisplayenv : "";
    if (waylanddisplay_str.empty()) {
      // check DISPLAY ?
      // const char* displayenv = getenv("DISPLAY");
      is_x11 = true;
    } else {
      is_wayland = true;
    }
  } else {
    is_wayland = (xdgsessiontype_str.find("wayland") != std::string::npos);
    is_x11 = (xdgsessiontype_str.find("x11") != std::string::npos);
  }
  WGPUSurfaceDescriptorFromWaylandSurface wgpustruct1;
  WGPUSurfaceDescriptorFromXcbWindow wgpustruct2;
  WGPUSurfaceDescriptorFromXlibWindow wgpustruct3;
  if (is_wayland) {
    // # todo: wayland seems to be broken right now
    wgpustruct1.display = display_id;
    wgpustruct1.surface = win_id;
    wgpustruct1.chain.next = nullptr;
    wgpustruct1.chain.sType = WGPUSType_SurfaceDescriptorFromWaylandSurface;
    surface_descriptor.nextInChain = (const WGPUChainedStruct*)(&wgpustruct1);
    LOG_INFO << "Wayland surface descriptor";
  } else if (is_xcb) {
    //# todo: xcb untested
    wgpustruct2.connection = display_id;
    wgpustruct2.window = *((uint32_t*)(&win_id));
    wgpustruct2.chain.next = nullptr;
    wgpustruct2.chain.sType = WGPUSType_SurfaceDescriptorFromXcbWindow;
    surface_descriptor.nextInChain = (const WGPUChainedStruct*)(&wgpustruct2);
    LOG_INFO << "XCB surface descriptor";
  } else {
    wgpustruct3.display = display_id;
    wgpustruct3.window = *((uint32_t*)(&win_id));
    //    wgpustruct3.window = static_cast<uint32_t>(win_id); //*((uint32_t*)(&win_id));
    wgpustruct3.chain.next = nullptr;
    wgpustruct3.chain.sType = WGPUSType_SurfaceDescriptorFromXlibWindow;
    surface_descriptor.nextInChain = (const WGPUChainedStruct*)(&wgpustruct3);
    LOG_INFO << "Xlib surface descriptor";
  }

#else
  throw("Cannot get surface id: unsupported platform.");
#endif

  WGPUSurface surface = wgpuInstanceCreateSurface(sInstance, &surface_descriptor);
  return surface;
}
