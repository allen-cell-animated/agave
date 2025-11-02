#pragma once

#include "glad/glad.h"

#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QSurfaceFormat>

// Conditional includes based on graphics backend
#ifdef AGAVE_USE_VULKAN
#include <vulkan/vulkan.h>
#include "graphicsVulkan/IVulkanRenderWindow.h"
#else
#include "graphics/IRenderWindow.h"
#endif

#include <map>
#include <memory>
#include <string>

#if defined(__APPLE__) || defined(_WIN32)
#define HAS_EGL false
#else
#define HAS_EGL true
#endif

struct ImageGpu;
class ImageXYZC;
class RenderSettings;

#ifdef AGAVE_USE_VULKAN
using IRenderWindowBase = IVulkanRenderWindow;
#else
using IRenderWindowBase = IRenderWindow;
#endif

typedef void* EGLContext; // Forward declaration from EGL.h.

class renderlib
{
public:
  // Initialize with appropriate graphics backend
  static int initialize(std::string assetPath, bool headless = false, bool listDevices = false, int selectedGpu = 0);
  static void clearGpuVolumeCache();
  static void cleanup();

  static std::string assetPath();

  // usage of this cache:
  // websocketserver:
  //   preloaded images will be cached at preload time and never deallocated
  //   images loaded by command will not be cached here.
  //   gpu cache will stay alive until shutdown
  // desktop gui:
  //   there is no preloading.
  //   gpu cache will be cleared every time a new volume is loaded.
  // if do_cache is false, caller is responsible for deallocating from gpu mem.
  // if do_cache is true, caller can be assured gpu mem will stay cached until an event that clears the cache
  // events that clear the cache:
  // websocketserver:
  //    shutdown
  // desktop:
  //    load new image or shutdown
  static std::shared_ptr<ImageGpu> imageAllocGPU(std::shared_ptr<ImageXYZC> image, bool do_cache = true);
  static void imageDeallocGPU(std::shared_ptr<ImageXYZC> image);

  static QSurfaceFormat getQSurfaceFormat(bool enableDebug = false);
  static QOpenGLContext* createOpenGLContext();

  enum RendererType
  {
    RendererType_Pathtrace,
    RendererType_Raymarch
  };
  // factory method for creating renderers - returns appropriate type based on compile-time backend
  static IRenderWindowBase* createRenderer(RendererType rendererType, RenderSettings* rs = nullptr);
  static RendererType stringToRendererType(std::string rendererTypeString);
  static std::string rendererTypeToString(RendererType rendererType);

  // Graphics backend information
  static std::string getGraphicsBackend();
  static bool isVulkanEnabled();

private:
  static std::map<std::shared_ptr<ImageXYZC>, std::shared_ptr<ImageGpu>> sGpuImageCache;
};

// Forward declarations for context classes
#ifdef AGAVE_USE_VULKAN
class HeadlessVKContext;
class RendererVKContext;
#else
class HeadlessGLContext;
class RendererGLContext;
#endif

// Context management classes - conditional compilation
#ifndef AGAVE_USE_VULKAN
class HeadlessGLContext
{
public:
  HeadlessGLContext();
  ~HeadlessGLContext();
  void makeCurrent();
  void doneCurrent();

private:
#if HAS_EGL
  EGLContext m_eglCtx;
#endif
};

// wrap a gl context intended to run on a separate thread
// or be moved from main thread and back
class RendererGLContext
{
public:
  RendererGLContext();
  ~RendererGLContext();
  void configure(QOpenGLContext* glContext = nullptr);
  void init();
  void destroy();

  void makeCurrent();
  void doneCurrent();

private:
  bool m_ownGLContext;
  // only one of the following two can be non-null
  HeadlessGLContext* m_eglContext;
  QOpenGLContext* m_glContext;
  QOffscreenSurface* m_surface;

  void initQOpenGLContext();
};
#else
// Vulkan context classes (stubs for now)
class HeadlessVKContext
{
public:
  HeadlessVKContext();
  ~HeadlessVKContext();
  void makeCurrent();
  void doneCurrent();

private:
  VkInstance m_instance;
  VkDevice m_device;
};

class RendererVKContext
{
public:
  RendererVKContext();
  ~RendererVKContext();
  void configure(VkDevice device = VK_NULL_HANDLE);
  void init();
  void destroy();

  void makeCurrent();
  void doneCurrent();

private:
  bool m_ownVKDevice;
  VkInstance m_instance;
  VkDevice m_device;
  VkQueue m_queue;
};
#endif