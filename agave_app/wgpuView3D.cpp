#include "wgpuView3D.h"

#include "QRenderSettings.h"
#include "ViewerState.h"

#include "renderlib/AppScene.h"
#include "renderlib/IRenderWindow.h"
#include "renderlib/ImageXYZC.h"
#include "renderlib/Logging.h"
#include "renderlib/RenderSettings.h"

#include <glm.h>

#include <QGuiApplication>
#include <QScreen>
#include <QWindow>
#include <QtGui/QMouseEvent>

#include <cmath>
#include <iostream>

// Only Microsoft issue warnings about correct behaviour...
#ifdef _MSVC_VER
#pragma warning(disable : 4351)
#endif

static void
request_adapter_callback(WGPURequestAdapterStatus status, WGPUAdapter received, const char* message, void* userdata)
{
  if (message) {
    LOG_INFO << "request adapter callback: " << message;
  }
  //  UNUSED(status);
  //  UNUSED(message);

  *(WGPUAdapter*)userdata = received;
}
static void
request_device_callback(WGPURequestDeviceStatus status, WGPUDevice received, const char* message, void* userdata)
{
  if (message) {
    LOG_INFO << "request device callback: " << message;
  }
  // UNUSED(status);
  // UNUSED(message);

  *(WGPUDevice*)userdata = received;
}
static void
handle_device_lost(WGPUDeviceLostReason reason, char const* message, void* userdata)
{
  LOG_INFO << "DEVICE LOST (" << reason << "): " << message;
  // UNUSED(userdata);
}

static void
handle_uncaptured_error(WGPUErrorType type, char const* message, void* userdata)
{
  // UNUSED(userdata);

  LOG_INFO << "UNCAPTURED ERROR (" << type << "): " << message;
}

static void
printAdapterFeatures(WGPUAdapter adapter)
{
  size_t count = wgpuAdapterEnumerateFeatures(adapter, NULL);
  WGPUFeatureName* features = (WGPUFeatureName*)malloc(count * sizeof(WGPUFeatureName));
  wgpuAdapterEnumerateFeatures(adapter, features);

  printf("[]WGPUFeatureName {\n");

  for (size_t i = 0; i < count; i++) {
    WGPUFeatureName feature = features[i];
    switch ((uint32_t)feature) {
      case WGPUFeatureName_DepthClipControl:
        printf("\tDepthClipControl\n");
        break;

      case WGPUFeatureName_Depth24UnormStencil8:
        printf("\tDepth24UnormStencil8\n");
        break;

      case WGPUFeatureName_Depth32FloatStencil8:
        printf("\tDepth32FloatStencil8\n");
        break;

      case WGPUFeatureName_TimestampQuery:
        printf("\tTimestampQuery\n");
        break;

      case WGPUFeatureName_PipelineStatisticsQuery:
        printf("\tPipelineStatisticsQuery\n");
        break;

      case WGPUFeatureName_TextureCompressionBC:
        printf("\tTextureCompressionBC\n");
        break;

      case WGPUFeatureName_TextureCompressionETC2:
        printf("\tTextureCompressionETC2\n");
        break;

      case WGPUFeatureName_TextureCompressionASTC:
        printf("\tTextureCompressionASTC\n");
        break;

      case WGPUFeatureName_IndirectFirstInstance:
        printf("\tIndirectFirstInstance\n");
        break;

      case WGPUNativeFeature_PUSH_CONSTANTS:
        printf("\tWGPUNativeFeature_PUSH_CONSTANTS\n");
        break;

      case WGPUNativeFeature_TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES:
        printf("\tWGPUNativeFeature_TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES\n");
        break;

      default:
        printf("\tUnknown=%d\n", feature);
    }
  }

  printf("}\n");

  free(features);
}

WgpuView3D::WgpuView3D(QCamera* cam, QRenderSettings* qrs, RenderSettings* rs, QWidget* parent)
  : QWidget(parent)
  , m_etimer()
  , m_lastPos(0, 0)
  , m_renderSettings(rs)
  , m_renderer()
  , m_qcamera(cam)
  , m_cameraController(cam, &m_CCamera)
  , m_qrendersettings(qrs)
  , m_rendererType(1)
  , m_initialized(false)
  , m_fakeHidden(false)
{
  setAttribute(Qt::WA_PaintOnScreen);
  setAutoFillBackground(false);

  // IMPORTANT this is where the QT gui container classes send their values down into the
  // CScene object. GUI updates --> QT Object Changed() --> cam->Changed() -->
  // WgpuView3D->OnUpdateCamera
  QObject::connect(cam, SIGNAL(Changed()), this, SLOT(OnUpdateCamera()));
  QObject::connect(qrs, SIGNAL(Changed()), this, SLOT(OnUpdateQRenderSettings()));
  QObject::connect(qrs, SIGNAL(ChangedRenderer(int)), this, SLOT(OnUpdateRenderer(int)));
}

void
WgpuView3D::initCameraFromImage(Scene* scene)
{
  // Tell the camera about the volume's bounding box
  m_CCamera.m_SceneBoundingBox.m_MinP = scene->m_boundingBox.GetMinP();
  m_CCamera.m_SceneBoundingBox.m_MaxP = scene->m_boundingBox.GetMaxP();
  // reposition to face image
  m_CCamera.SetViewMode(ViewModeFront);

  RenderSettings& rs = *m_renderSettings;
  rs.m_DirtyFlags.SetFlag(CameraDirty);
}

void
WgpuView3D::toggleCameraProjection()
{
  ProjectionMode p = m_CCamera.m_Projection;
  m_CCamera.SetProjectionMode((p == PERSPECTIVE) ? ORTHOGRAPHIC : PERSPECTIVE);

  RenderSettings& rs = *m_renderSettings;
  rs.m_DirtyFlags.SetFlag(CameraDirty);
}

void
WgpuView3D::onNewImage(Scene* scene)
{
  m_renderer->setScene(scene);
  // costly teardown and rebuild.
  this->OnUpdateRenderer(m_rendererType);
  // would be better to preserve renderer and just change the scene data to include the new image.
  // how tightly coupled is renderer and scene????
}

WgpuView3D::~WgpuView3D() {}

QSize
WgpuView3D::minimumSizeHint() const
{
  return QSize(800, 600);
}

QSize
WgpuView3D::sizeHint() const
{
  return QSize(800, 600);
}

void
WgpuView3D::initializeGL()
{
  if (m_initialized) {
    return;
  }
  m_surface = renderlib_wgpu::get_surface_id_from_canvas((void*)winId());
  WGPUAdapter adapter;
  WGPURequestAdapterOptions options = {
    .nextInChain = NULL,
    .compatibleSurface = m_surface,
  };
  wgpuInstanceRequestAdapter(NULL, &options, request_adapter_callback, (void*)&adapter);
  printAdapterFeatures(adapter);

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
  WGPUDeviceDescriptor deviceDescriptor = {
    .nextInChain = (const WGPUChainedStruct*)&deviceExtras,
    .requiredLimits = &requiredLimits,
    .defaultQueue =
      WGPUQueueDescriptor{
        .nextInChain = NULL,
        .label = NULL,
      },
  };

  // creates/ fills in m_device!
  wgpuAdapterRequestDevice(adapter, &deviceDescriptor, request_device_callback, (void*)&m_device);

  wgpuDeviceSetUncapturedErrorCallback(m_device, handle_uncaptured_error, NULL);
  wgpuDeviceSetDeviceLostCallback(m_device, handle_device_lost, NULL);

  m_swapChainFormat = wgpuSurfaceGetPreferredFormat(m_surface, adapter);
  WGPUSwapChainDescriptor swapChainDescriptor = {
    .usage = WGPUTextureUsage_RenderAttachment,
    .format = m_swapChainFormat,
    .width = (uint32_t)width(),
    .height = (uint32_t)height(),
    .presentMode = WGPUPresentMode_Fifo,
  };
  // need to do this on resize
  m_swapChain = wgpuDeviceCreateSwapChain(m_device, m_surface, &swapChainDescriptor);

  // The WgpuView3D owns one CScene

  m_cameraController.setRenderSettings(*m_renderSettings);
  m_qrendersettings->setRenderSettings(*m_renderSettings);

  WGPUPipelineLayoutDescriptor pipelineLayoutDescriptor = { .bindGroupLayoutCount = 0, .bindGroupLayouts = NULL };
  WGPUPipelineLayout pipelineLayout = wgpuDeviceCreatePipelineLayout(m_device, &pipelineLayoutDescriptor);
  WGPUBlendState blendState = { .color =
                                  WGPUBlendComponent{
                                    .operation = WGPUBlendOperation_Add,
                                    .srcFactor = WGPUBlendFactor_One,
                                    .dstFactor = WGPUBlendFactor_Zero,
                                  },
                                .alpha = WGPUBlendComponent{
                                  .operation = WGPUBlendOperation_Add,
                                  .srcFactor = WGPUBlendFactor_One,
                                  .dstFactor = WGPUBlendFactor_Zero,
                                } };

  WGPUColorTargetState colorTargetState = { .format = m_swapChainFormat,
                                            .blend = &blendState,
                                            .writeMask = WGPUColorWriteMask_All };
  WGPUFragmentState fragmentState = {
    .module = nullptr, // shader,
    .entryPoint = "fs_main",
    .targetCount = 1,
    .targets = &colorTargetState,
  };

  WGPURenderPipelineDescriptor renderPipelineDescriptor = {
    .label = "Render pipeline",
    .layout = pipelineLayout,
    .vertex =
      WGPUVertexState{
        .module = nullptr,     // shader,
        .entryPoint = nullptr, //"",  //"vs_main",
        .bufferCount = 0,
        .buffers = NULL,
      },
    .primitive = WGPUPrimitiveState{ .topology = WGPUPrimitiveTopology_TriangleList,
                                     .stripIndexFormat = WGPUIndexFormat_Undefined,
                                     .frontFace = WGPUFrontFace_CCW,
                                     .cullMode = WGPUCullMode_None },
    .depthStencil = NULL,
    .multisample =
      WGPUMultisampleState{
        .count = 1,
        .mask = (uint32_t)~0,
        .alphaToCoverageEnabled = false,
      },
    .fragment = nullptr, //&fragmentState,
  };

  // m_pipeline = wgpuDeviceCreateRenderPipeline(m_device, &renderPipelineDescriptor);
  m_initialized = true;
  if (!m_renderer) {
    return;
  }
  QSize newsize = size();
  m_renderer->initialize(newsize.width(), newsize.height(), devicePixelRatioF());

  // Start timers
  startTimer(0);
  m_etimer.start();

  // // Size viewport
  // resizeGL(newsize.width(), newsize.height());
}

void
WgpuView3D::paintEvent(QPaintEvent* e)
{
  Q_UNUSED(e);
  if (!m_initialized)
    return;
  if (updatesEnabled())
    render();
}

void
WgpuView3D::render()
{
  if (m_fakeHidden || !m_initialized) {
    return;
  }

  QWindow* win = windowHandle();
  if (!win || !win->isExposed())
    return;
  WGPUTextureView nextTexture = wgpuSwapChainGetCurrentTextureView(m_swapChain);
  if (!nextTexture) {
    // try one time to re-create swap chain
    WGPUSwapChainDescriptor swapChainDescriptor = {
      .usage = WGPUTextureUsage_RenderAttachment,
      .format = m_swapChainFormat,
      .width = (uint32_t)width(),
      .height = (uint32_t)height(),
      .presentMode = WGPUPresentMode_Fifo,
    };
    m_swapChain = wgpuDeviceCreateSwapChain(m_device, m_surface, &swapChainDescriptor);
    nextTexture = wgpuSwapChainGetCurrentTextureView(m_swapChain);
  }
  invokeUserPaint(nextTexture);

  wgpuSwapChainPresent(m_swapChain);
}

void
WgpuView3D::invokeUserPaint(WGPUTextureView nextTexture)
{
  paintGL(nextTexture);
  // flush? (queue submit?)
}

void
WgpuView3D::paintGL(WGPUTextureView nextTexture)
{
  WGPUCommandEncoderDescriptor commandEncoderDescriptor = { .label = "Command Encoder" };
  WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(m_device, &commandEncoderDescriptor);
  WGPURenderPassColorAttachment renderPassColorAttachment = {
    .view = nextTexture,
    .resolveTarget = 0,
    .loadOp = WGPULoadOp_Clear,
    .storeOp = WGPUStoreOp_Store,
    .clearValue =
      WGPUColor{
        .r = 0.0,
        .g = 1.0,
        .b = 0.0,
        .a = 1.0,
      },
  };
  WGPURenderPassDescriptor renderPassDescriptor = {
    .colorAttachmentCount = 1,
    .colorAttachments = &renderPassColorAttachment,
    .depthStencilAttachment = NULL,
  };
  WGPURenderPassEncoder renderPass = wgpuCommandEncoderBeginRenderPass(encoder, &renderPassDescriptor);

  // wgpuRenderPassEncoderSetPipeline(renderPass, pipeline);
  // wgpuRenderPassEncoderDraw(renderPass, 3, 1, 0, 0);
  wgpuRenderPassEncoderEnd(renderPass);
  wgpuTextureViewDrop(nextTexture);

  WGPUQueue queue = wgpuDeviceGetQueue(m_device);
  WGPUCommandBufferDescriptor commandBufferDescriptor = { .label = NULL };
  WGPUCommandBuffer cmdBuffer = wgpuCommandEncoderFinish(encoder, &commandBufferDescriptor);
  wgpuQueueSubmit(queue, 1, &cmdBuffer);

  if (!m_renderer) {
    return;
  }
  m_CCamera.Update();

  m_renderer->render(m_CCamera);
}

void
WgpuView3D::resizeEvent(QResizeEvent* event)
{
  if (event->size().isEmpty()) {
    m_fakeHidden = true;
    return;
  }
  m_fakeHidden = false;
  initializeGL();
  if (!m_initialized) {
    return;
  }

  int w = event->size().width();
  int h = event->size().height();

  m_CCamera.m_Film.m_Resolution.SetResX(w);
  m_CCamera.m_Film.m_Resolution.SetResY(h);

  // (if w or h actually changed...)
  WGPUSwapChainDescriptor swapChainDescriptor = {
    .usage = WGPUTextureUsage_RenderAttachment,
    .format = m_swapChainFormat,
    .width = (uint32_t)w,
    .height = (uint32_t)h,
    .presentMode = WGPUPresentMode_Fifo,
  };
  m_swapChain = wgpuDeviceCreateSwapChain(m_device, m_surface, &swapChainDescriptor);

  if (m_renderer) {
    m_renderer->resize(w, h, devicePixelRatioF());
  }

  // invokeUserPaint();
}

void
WgpuView3D::mousePressEvent(QMouseEvent* event)
{
  m_lastPos = event->pos();
  m_cameraController.m_OldPos[0] = m_lastPos.x();
  m_cameraController.m_OldPos[1] = m_lastPos.y();
}

void
WgpuView3D::mouseReleaseEvent(QMouseEvent* event)
{
  m_lastPos = event->pos();
  m_cameraController.m_OldPos[0] = m_lastPos.x();
  m_cameraController.m_OldPos[1] = m_lastPos.y();
}

// No switch default to avoid -Wunreachable-code errors.
// However, this then makes -Wswitch-default complain.  Disable
// temporarily.
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-default"
#endif

// x, y in 0..1 relative to screen
static glm::vec3
get_arcball_vector(float xndc, float yndc)
{
  glm::vec3 P = glm::vec3(1.0 * xndc * 2 - 1.0, 1.0 * yndc * 2 - 1.0, 0);
  P.y = -P.y;
  float OP_squared = P.x * P.x + P.y * P.y;
  if (OP_squared <= 1 * 1)
    P.z = sqrt(1 * 1 - OP_squared); // Pythagore
  else
    P = glm::normalize(P); // nearest point
  return P;
}

void
WgpuView3D::mouseMoveEvent(QMouseEvent* event)
{
  m_cameraController.OnMouseMove(event);
  m_lastPos = event->pos();
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

void
WgpuView3D::timerEvent(QTimerEvent* event)
{

  QWidget::timerEvent(event);

  update();
}

void
WgpuView3D::OnUpdateCamera()
{
  //	QMutexLocker Locker(&gSceneMutex);
  RenderSettings& rs = *m_renderSettings;
  m_CCamera.m_Film.m_Exposure = 1.0f - m_qcamera->GetFilm().GetExposure();
  m_CCamera.m_Film.m_ExposureIterations = m_qcamera->GetFilm().GetExposureIterations();

  if (m_qcamera->GetFilm().IsDirty()) {
    const int FilmWidth = m_qcamera->GetFilm().GetWidth();
    const int FilmHeight = m_qcamera->GetFilm().GetHeight();

    m_CCamera.m_Film.m_Resolution.SetResX(FilmWidth);
    m_CCamera.m_Film.m_Resolution.SetResY(FilmHeight);
    m_CCamera.Update();
    m_qcamera->GetFilm().UnDirty();

    rs.m_DirtyFlags.SetFlag(FilmResolutionDirty);
  }

  m_CCamera.Update();

  // Aperture
  m_CCamera.m_Aperture.m_Size = m_qcamera->GetAperture().GetSize();

  // Projection
  m_CCamera.m_FovV = m_qcamera->GetProjection().GetFieldOfView();

  // Focus
  m_CCamera.m_Focus.m_Type = (Focus::EType)m_qcamera->GetFocus().GetType();
  m_CCamera.m_Focus.m_FocalDistance = m_qcamera->GetFocus().GetFocalDistance();

  rs.m_DenoiseParams.m_Enabled = m_qcamera->GetFilm().GetNoiseReduction();

  rs.m_DirtyFlags.SetFlag(CameraDirty);
}
void
WgpuView3D::OnUpdateQRenderSettings(void)
{
  // QMutexLocker Locker(&gSceneMutex);
  RenderSettings& rs = *m_renderSettings;

  rs.m_RenderSettings.m_DensityScale = m_qrendersettings->GetDensityScale();
  rs.m_RenderSettings.m_ShadingType = m_qrendersettings->GetShadingType();
  rs.m_RenderSettings.m_GradientFactor = m_qrendersettings->GetGradientFactor();

  // update window/levels / transfer function here!!!!

  rs.m_DirtyFlags.SetFlag(TransferFunctionDirty);
}

std::shared_ptr<CStatus>
WgpuView3D::getStatus()
{
  return m_renderer->getStatusInterface();
}

void
WgpuView3D::OnUpdateRenderer(int rendererType)
{
#if 0
  // clean up old renderer.
  if (m_renderer) {
    m_renderer->cleanUpResources();
  }

  Scene* sc = m_renderer->scene();

  switch (rendererType) {
    case 1:
      LOG_DEBUG << "Set OpenGL pathtrace Renderer";
      m_renderer.reset(new RenderGLPT(m_renderSettings));
      m_renderSettings->m_DirtyFlags.SetFlag(TransferFunctionDirty);
      break;
    case 2:
      LOG_DEBUG << "Set OpenGL pathtrace Renderer";
      m_renderer.reset(new RenderGLPT(m_renderSettings));
      m_renderSettings->m_DirtyFlags.SetFlag(TransferFunctionDirty);
      break;
    default:
      LOG_DEBUG << "Set OpenGL single pass Renderer";
      m_renderer.reset(new RenderGL(m_renderSettings));
  };
  m_rendererType = rendererType;

  QSize newsize = size();
  // need to update the scene in QAppearanceSettingsWidget.
  m_renderer->setScene(sc);
  m_renderer->initialize(newsize.width(), newsize.height(), devicePixelRatioF());

  m_renderSettings->m_DirtyFlags.SetFlag(RenderParamsDirty);

  emit ChangedRenderer();
#endif
}

void
WgpuView3D::fromViewerState(const Serialize::ViewerState& s)
{
  m_CCamera.m_From = glm::vec3(s.m_eyeX, s.m_eyeY, s.m_eyeZ);
  m_CCamera.m_Target = glm::vec3(s.m_targetX, s.m_targetY, s.m_targetZ);
  m_CCamera.m_Up = glm::vec3(s.m_upX, s.m_upY, s.m_upZ);
  m_CCamera.m_FovV = s.m_fov;
  m_CCamera.SetProjectionMode(s.m_projection == ViewerState::Projection::PERSPECTIVE ? PERSPECTIVE : ORTHOGRAPHIC);
  m_CCamera.m_OrthoScale = s.m_orthoScale;

  m_CCamera.m_Film.m_Exposure = s.m_exposure;
  m_CCamera.m_Aperture.m_Size = s.m_apertureSize;
  m_CCamera.m_Focus.m_FocalDistance = s.m_focalDistance;

  // TODO disentangle these QCamera* _camera and CCamera mCamera objects. Only CCamera should be necessary, I think.
  m_qcamera->GetProjection().SetFieldOfView(s.m_fov);
  m_qcamera->GetFilm().SetExposure(s.m_exposure);
  m_qcamera->GetAperture().SetSize(s.m_apertureSize);
  m_qcamera->GetFocus().SetFocalDistance(s.m_focalDistance);
}

QPixmap
WgpuView3D::capture()
{
  // get the current QScreen
  QScreen* screen = QGuiApplication::primaryScreen();
  if (const QWindow* window = windowHandle()) {
    screen = window->screen();
  }
  if (!screen) {
    qWarning("Couldn't capture screen to save image file.");
    return QPixmap();
  }
  // simply grab the glview window
  return screen->grabWindow(winId());
}

QImage
WgpuView3D::captureQimage()
{
#if 0
  makeCurrent();

  // Create a one-time FBO to receive the image
  QOpenGLFramebufferObjectFormat fboFormat;
  fboFormat.setAttachment(QOpenGLFramebufferObject::NoAttachment);
  fboFormat.setMipmap(false);
  fboFormat.setSamples(0);
  fboFormat.setTextureTarget(GL_TEXTURE_2D);

  // NOTE NO ALPHA. if alpha then this will get premultiplied and wash out colors
  // TODO : allow user option for transparent qimage, and then put GL_RGBA8 back here
  fboFormat.setInternalTextureFormat(GL_RGB8);
  check_gl("pre screen capture");

  QOpenGLFramebufferObject* fbo =
    new QOpenGLFramebufferObject(width() * devicePixelRatioF(), height() * devicePixelRatioF(), fboFormat);
  check_gl("create fbo");

  fbo->bind();
  check_glfb("bind framebuffer for screen capture");

  // do a render into the temp framebuffer
  glViewport(0, 0, fbo->width(), fbo->height());
  m_renderer->render(m_CCamera);
  fbo->release();

  QImage img(fbo->toImage());
  delete fbo;

  return img;
#endif
  return QImage();
}

void
WgpuView3D::pauseRenderLoop()
{
  std::shared_ptr<CStatus> s = getStatus();
  // the CStatus updates can cause Qt GUI work to happen,
  // which can not be called from a separate thread.
  // so when we start rendering from another thread,
  // we need to either make status updates thread safe,
  // or just disable them here.
  s->EnableUpdates(false);
  m_etimer.invalidate();
}

void
WgpuView3D::restartRenderLoop()
{
  m_etimer.restart();
  std::shared_ptr<CStatus> s = getStatus();
  s->EnableUpdates(true);
}
