#pragma once

#include "glad/glad.h"

#include <QOffscreenSurface>
#include <QSurfaceFormat>
#include <QtGui/QOpenGLContext>
#include <QtGui/QOpenGLDebugLogger>
#include <QtGui/QWindow>

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

typedef void* EGLContext; // Forward declaration from EGL.h.

class renderlib
{
public:
  static int initialize(bool headless = false, bool listDevices = false, int selectedGpu = 0);
  static void clearGpuVolumeCache();
  static void cleanup();

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

private:
  static std::map<std::shared_ptr<ImageXYZC>, std::shared_ptr<ImageGpu>> sGpuImageCache;
};

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