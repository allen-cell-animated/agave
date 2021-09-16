#include "renderlib.h"

#include "ImageXYZC.h"
#include "ImageXyzcGpu.h"
#include "Logging.h"

#include <string>

#if HAS_EGL
#include <EGL/egl.h>
#include <EGL/eglext.h>
#endif

static bool renderLibInitialized = false;

static bool renderLibHeadless = false;
#if HAS_EGL
static EGLDisplay eglDpy = NULL;
#endif

static QOpenGLContext* dummyContext = nullptr;
static QOffscreenSurface* dummySurface = nullptr;

static QOpenGLDebugLogger* logger = nullptr;

std::map<std::shared_ptr<ImageXYZC>, std::shared_ptr<ImageGpu>> renderlib::sGpuImageCache;

static const struct
{
  int major = 4;
  int minor = 1;
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

QOpenGLContext*
renderlib::createOpenGLContext()
{
  QOpenGLContext* context = new QOpenGLContext();
  context->setFormat(getQSurfaceFormat()); // ...and set the format on the context too

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

#if HAS_EGL

void
checkEGLError(std::string message)
{
  EGLint lastError = EGL_SUCCESS;
  if ((lastError = eglGetError()) != EGL_SUCCESS) {
    LOG_ERROR << "eglGetError " << lastError;
    LOG_ERROR << message;
  }
}

EGLDisplay
getEGLDefaultDisplay()
{
  EGLint lastError = EGL_SUCCESS;
  EGLDisplay eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  LOG_INFO << "eglGetDisplay returns " << eglDpy;
  checkEGLError("Failed eglGetDisplay");
  return eglDisplay;
}

EGLDisplay
initEGLDisplay()
{
  PFNEGLQUERYDEVICESEXTPROC eglQueryDevicesEXT = (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
  checkEGLError("Failed to get EGLEXT: eglQueryDevicesEXT");
  PFNEGLGETPLATFORMDISPLAYEXTPROC eglGetPlatformDisplayEXT =
    (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
  checkEGLError("Failed to get EGLEXT: eglGetPlatformDisplayEXT");
  PFNEGLQUERYDEVICEATTRIBEXTPROC eglQueryDeviceAttribEXT =
    (PFNEGLQUERYDEVICEATTRIBEXTPROC)eglGetProcAddress("eglQueryDeviceAttribEXT");
  checkEGLError("Failed to get EGLEXT: eglQueryDeviceAttribEXT");
  PFNEGLQUERYDEVICESTRINGEXTPROC eglQueryDeviceStringEXT =
    (PFNEGLQUERYDEVICESTRINGEXTPROC)eglGetProcAddress("eglQueryDeviceStringEXT");
  checkEGLError("Failed to get EGLEXT: eglQueryDeviceStringEXT");

  if (!eglQueryDevicesEXT || !eglGetPlatformDisplayEXT || !eglQueryDeviceAttribEXT || !eglQueryDeviceStringEXT) {
    return getEGLDefaultDisplay();
  }

  EGLint numberDevices;
  // Get number of devices
  EGLBoolean ok = eglQueryDevicesEXT(0, NULL, &numberDevices);
  if (!ok) {
    LOG_ERROR << "Failed to get number of devices. Bad parameter suspected";
  }
  checkEGLError("Error getting number of devices: eglQueryDevicesEXT");

  LOG_INFO << numberDevices << " devices found";
  if (numberDevices > 0) {
    EGLDeviceEXT* eglDevs = new EGLDeviceEXT[numberDevices];
    ok = eglQueryDevicesEXT(numberDevices, eglDevs, &numberDevices);
    if (!ok) {
      LOG_ERROR << "Failed to get devices. Bad parameter suspected";
    }
    checkEGLError("Error getting number of devices: eglQueryDevicesEXT");
    for (int i = 0; i < numberDevices; ++i) {
      LOG_INFO << "Device " << i << ":";
      const char* vendorstring = eglQueryDeviceStringEXT(eglDevs[i], EGL_VENDOR);
      checkEGLError("Error retreiving EGL_VENDOR string for device");
      LOG_INFO << "  " << vendorstring;
      const char* extensionsstring = eglQueryDeviceStringEXT(eglDevs[i], EGL_EXTENSIONS);
      checkEGLError("Error retreiving EGL_EXTENSIONS string for device");
      LOG_INFO << "  " << extensionsstring;
    }
    // select device by index
    EGLDisplay eglDisplay = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, eglDevs[0], 0);
    checkEGLError("Error getting Platform Display: eglGetPlatformDisplayEXT");
    return eglDisplay;
  } else {
    return getEGLDefaultDisplay();
  }
}
#endif

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

  HeadlessGLContext* dummyHeadlessContext = nullptr;

  if (headless) {
#if HAS_EGL

    // one-time EGL init

    EGLint lastError = EGL_SUCCESS;

    // 1. Initialize EGL
    eglDpy = initEGLDisplay();

    EGLint major, minor;

    EGLBoolean init_ok = eglInitialize(eglDpy, &major, &minor);
    if (init_ok == EGL_FALSE) {
      LOG_ERROR << "renderlib::initialize, eglInitialize failed";
    }
    if ((lastError = eglGetError()) != EGL_SUCCESS) {
      LOG_ERROR << "eglGetError " << lastError;
    }
    // 2. Bind the API
    EGLBoolean bindapi_ok = eglBindAPI(EGL_OPENGL_API);
    if (bindapi_ok == EGL_FALSE) {
      LOG_ERROR << "renderlib::initialize, eglBindAPI failed";
    }
    if ((lastError = eglGetError()) != EGL_SUCCESS) {
      LOG_ERROR << "eglGetError " << lastError;
    }
    dummyHeadlessContext = new HeadlessGLContext();
    dummyHeadlessContext->makeCurrent();
#else
    LOG_ERROR << "Headless operation without EGL support is not available";
#endif
  } else {
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
  }

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

  delete dummyHeadlessContext;
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

HeadlessGLContext::HeadlessGLContext()
{
#if HAS_EGL
  EGLint lastError = EGL_SUCCESS;

  // Bind the API
  EGLBoolean bindapi_ok = eglBindAPI(EGL_OPENGL_API);
  if (bindapi_ok == EGL_FALSE) {
    LOG_ERROR << "renderlib::initialize, eglBindAPI failed";
  }
  if ((lastError = eglGetError()) != EGL_SUCCESS) {
    LOG_ERROR << "eglGetError " << lastError;
  }

  // Select an appropriate configuration
  EGLint numConfigs;
  EGLConfig eglCfg;

  static const EGLint configAttribs[] = { EGL_SURFACE_TYPE,
                                          EGL_PBUFFER_BIT,
                                          EGL_ALPHA_SIZE,
                                          8,
                                          EGL_BLUE_SIZE,
                                          8,
                                          EGL_GREEN_SIZE,
                                          8,
                                          EGL_RED_SIZE,
                                          8,
                                          EGL_DEPTH_SIZE,
                                          AICS_DEFAULT_DEPTH_BUFFER_BITS,
                                          EGL_STENCIL_SIZE,
                                          AICS_DEFAULT_STENCIL_BUFFER_BITS,
                                          EGL_RENDERABLE_TYPE,
                                          EGL_OPENGL_BIT,
                                          EGL_NONE };
  EGLBoolean chooseConfig_ok = eglChooseConfig(eglDpy, configAttribs, &eglCfg, 1, &numConfigs);
  if (chooseConfig_ok == EGL_FALSE) {
    LOG_ERROR << "renderlib::initialize, eglChooseConfig failed";
  }
  if ((lastError = eglGetError()) != EGL_SUCCESS) {
    LOG_ERROR << "eglGetError " << lastError;
  }

  // Create a context and make it current
  static const EGLint contextAttribs[] = {
    EGL_CONTEXT_MAJOR_VERSION, AICS_GL_VERSION.major, EGL_CONTEXT_MINOR_VERSION, AICS_GL_VERSION.minor, EGL_NONE
  };
  EGLContext eglCtx = eglCreateContext(eglDpy, eglCfg, EGL_NO_CONTEXT, contextAttribs);
  if (eglCtx == EGL_NO_CONTEXT) {
    LOG_ERROR << "renderlib::initialize, eglCreateContext failed";
  } else {
    LOG_INFO << "created a egl context";
  }
  if ((lastError = eglGetError()) != EGL_SUCCESS) {
    LOG_ERROR << "eglGetError " << lastError;
  }

  m_eglCtx = eglCtx;
#endif
}

HeadlessGLContext::~HeadlessGLContext()
{
#if HAS_EGL
  eglDestroyContext(eglDpy, m_eglCtx);
#endif
}

void
HeadlessGLContext::makeCurrent()
{
#if HAS_EGL
  LOG_INFO << "pre-eglMakeCurrent";
  eglMakeCurrent(eglDpy, EGL_NO_SURFACE, EGL_NO_SURFACE, m_eglCtx);
  LOG_INFO << "post-eglMakeCurrent";
#endif
}

void
HeadlessGLContext::doneCurrent()
{
#if HAS_EGL
  eglMakeCurrent(eglDpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
#endif
}
