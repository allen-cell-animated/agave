#include "renderlib.h"

#include "ImageXYZC.h"
#include "ImageXyzcGpu.h"
#include "Logging.h"
#include "glad/glad.h"

#include <string>

#include <QOffscreenSurface>
#include <QtGui/QOpenGLContext>
#include <QtGui/QOpenGLDebugLogger>
#include <QtGui/QWindow>

#if defined(__APPLE__) || defined(_WIN32)
#define HAS_EGL false
#else
#define HAS_EGL true
#endif

#if HAS_EGL
#include <EGL/egl.h>
#include <QtPlatformHeaders/QEGLNativeContext>
#endif

static bool renderLibInitialized = false;

static bool renderLibHeadless = false;
#if HAS_EGL
static EGLDisplay eglDpy = NULL;
static EGLContext eglCtx = NULL;
#endif

static QOpenGLContext* dummyContext = nullptr;
static QOffscreenSurface* dummySurface = nullptr;

static QOpenGLDebugLogger* logger = nullptr;

std::map<std::shared_ptr<ImageXYZC>, std::shared_ptr<ImageGpu>> renderlib::sGpuImageCache;

static const struct
{
  int major = 3;
  int minor = 3;
} AICS_GL_VERSION;

static const uint32_t AICS_DEFAULT_STENCIL_BUFFER_BITS = 8;

static const uint32_t AICS_DEFAULT_DEPTH_BUFFER_BITS = 24;

namespace {
static void
logMessage(QOpenGLDebugMessage message)
{
  LOG_DEBUG << message.message().toStdString();
}
}

QSurfaceFormat
renderlib::getQSurfaceFormat(bool enableDebug)
{
  QSurfaceFormat format;
  format.setDepthBufferSize(AICS_DEFAULT_DEPTH_BUFFER_BITS);
  format.setStencilBufferSize(AICS_DEFAULT_STENCIL_BUFFER_BITS);
  format.setVersion(AICS_GL_VERSION.major, AICS_GL_VERSION.minor);
  // necessary on MacOS at least:
  format.setProfile(QSurfaceFormat::CoreProfile);
  if (enableDebug) {
    format.setOption(QSurfaceFormat::DebugContext);
  }
  return format;
}

QOpenGLContext* renderlib::createOpenGLContext() {
  QOpenGLContext* context = new QOpenGLContext();

  if (renderLibHeadless) {
    #if HAS_EGL
    context->setNativeHandle(QVariant::fromValue(QEGLNativeContext(eglCtx, eglDpy)));
    #endif
    context->setFormat(getQSurfaceFormat()); // ...and set the format on the context too
  }
  else {
    context->setFormat(getQSurfaceFormat()); // ...and set the format on the context too
  }
  bool createdOk = context->create();
  if (!createdOk) {
    LOG_ERROR << "Failed to create OpenGL Context";
  } else {
    LOG_INFO << "Created opengl context";
  }
  if (!context->isValid()) {
    LOG_ERROR << "Created GL Context is not valid";
  }

  return context;
}

int
renderlib::initialize(bool headless)
{
  if (renderLibInitialized) {
    return 1;
  }
  renderLibInitialized = true;

  // no MACOS support for EGL
  #if HAS_EGL
  #else
  headless = false;
  #endif
  renderLibHeadless = headless;

  // boost::log::add_file_log("renderlib.log");
  LOG_INFO << "Renderlib startup";

  bool enableDebug = false;

  QSurfaceFormat format = getQSurfaceFormat();
  QSurfaceFormat::setDefaultFormat(format);

  if (headless) {
    #if HAS_EGL

    // 1. Initialize EGL
    eglDpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    EGLint major, minor;

    EGLBoolean init_ok = eglInitialize(eglDpy, &major, &minor);
    if(init_ok == EGL_FALSE) {
      LOG_ERROR << "renderlib::initialize, eglInitialize failed";
    }

    // 2. Select an appropriate configuration
    EGLint numConfigs;
    EGLConfig eglCfg;

    static const EGLint configAttribs[] = {
              EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
              EGL_ALPHA_SIZE, 8,
              EGL_BLUE_SIZE, 8,
              EGL_GREEN_SIZE, 8,
              EGL_RED_SIZE, 8,
              EGL_DEPTH_SIZE, 24,
              EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
              EGL_NONE
    };
    EGLBoolean chooseConfig_ok = eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);
    if (chooseConfig_ok == EGL_FALSE) {
      LOG_ERROR << "renderlib::initialize, eglChooseConfig failed";
    }

    // 3. create a surface (SKIPPING)

    // 4. Bind the API
    EGLBoolean bindapi_ok = eglBindAPI(EGL_OPENGL_API);
    if(bindapi_ok == EGL_FALSE) {
      LOG_ERROR << "renderlib::initialize, eglBindAPI failed";
    }

    // 5. Create a context and make it current
    static const EGLint contextAttribs[] = {
              EGL_CONTEXT_MAJOR_VERSION, AICS_GL_VERSION.major,
              EGL_CONTEXT_MINOR_VERSION, AICS_GL_VERSION.minor,
              EGL_NONE
    };
    eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT,
                                        contextAttribs);
    if (eglCtx == EGL_NO_CONTEXT) {
      LOG_ERROR << "renderlib::initialize, eglCreateContext failed";
    }
    else {
      LOG_INFO << "created a egl context";
    }
    #endif
  }

  dummyContext = renderlib::createOpenGLContext();

  dummySurface = new QOffscreenSurface();
  dummySurface->setFormat(dummyContext->format());
  dummySurface->create();
  LOG_INFO << "Created offscreen surface";
  if (!dummySurface->isValid()) {
    LOG_ERROR << "QOffscreenSurface is not valid";
  }
  bool ok = dummyContext->makeCurrent(dummySurface);
  if (!ok) {
    LOG_ERROR << "Failed to makeCurrent on offscreen surface";
  } else {
    LOG_INFO << "Made context current on offscreen surface";
  }

  //	dummyWidget = new QOpenGLWidget();
  //	dummyWidget->setMaximumSize(2, 2);
  //	dummyWidget->show();
  //	dummyWidget->hide();
  //	dummyWidget->makeCurrent();

  if (enableDebug) {
    logger = new QOpenGLDebugLogger();
    QObject::connect(logger, &QOpenGLDebugLogger::messageLogged, logMessage);
    if (logger->initialize()) {
      logger->startLogging(QOpenGLDebugLogger::SynchronousLogging);
      logger->enableMessages();
    }
  }

  // note: there MUST be a valid current gl context in order to run this:
  int status = gladLoadGL();
  if (!status) {
    LOG_ERROR << "Failed to init GL";
    return status;
  }

  LOG_INFO << "GL_VENDOR: " << std::string((char*)glGetString(GL_VENDOR));
  LOG_INFO << "GL_RENDERER: " << std::string((char*)glGetString(GL_RENDERER));

  return status;
}

void
renderlib::clearGpuVolumeCache()
{
  // clean up the shared gpu buffer cache
  for (auto i : sGpuImageCache) {
    i.second->deallocGpu();
  }
  sGpuImageCache.clear();
}

void
renderlib::cleanup()
{
  if (!renderLibInitialized) {
    return;
  }
  LOG_INFO << "Renderlib shutdown";

  clearGpuVolumeCache();

  delete dummySurface;
  dummySurface = nullptr;
  delete dummyContext;
  dummyContext = nullptr;
  delete logger;
  logger = nullptr;

  if (renderLibHeadless) {
    #if HAS_EGL
    eglTerminate(eglDpy);
    #endif
  }
  renderLibInitialized = false;
}

std::shared_ptr<ImageGpu>
renderlib::imageAllocGPU(std::shared_ptr<ImageXYZC> image, bool do_cache)
{
  auto cached = sGpuImageCache.find(image);
  if (cached != sGpuImageCache.end()) {
    return cached->second;
  }

  ImageGpu* cimg = new ImageGpu;
  cimg->allocGpuInterleaved(image.get());
  std::shared_ptr<ImageGpu> shared(cimg);

  if (do_cache) {
    sGpuImageCache[image] = shared;
  }

  return shared;
}

void
renderlib::imageDeallocGPU(std::shared_ptr<ImageXYZC> image)
{
  auto cached = sGpuImageCache.find(image);
  if (cached != sGpuImageCache.end()) {
    // cached->second is a ImageGpu.
    // outstanding shared refs to cached->second will be deallocated!?!?!?!
    cached->second->deallocGpu();
    sGpuImageCache.erase(image);
  }
}
