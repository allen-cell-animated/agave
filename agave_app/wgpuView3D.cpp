#include "wgpuView3D.h"

#include "Camera.h"
#include "QRenderSettings.h"
#include "ViewerState.h"

#include "renderlib/AppScene.h"
#include "renderlib/ImageXYZC.h"
#include "renderlib/Logging.h"
#include "renderlib/MoveTool.h"
#include "renderlib/RenderSettings.h"
#include "renderlib/RotateTool.h"
#include "renderlib/graphics/IRenderWindow.h"
#include "renderlib/graphics/RenderGL.h"
#include "renderlib/graphics/RenderGLPT.h"
#include "renderlib/graphics/gl/Image3D.h"
#include "renderlib/graphics/gl/Util.h"
#include "renderlib_wgpu/getsurface_wgpu.h"
#include "renderlib_wgpu/wgpu_util.h"

#include <glm.h>

#include <QApplication>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QResizeEvent>
#include <QScreen>
#include <QTimer>
#include <QWindow>

#include <cmath>
#include <iostream>

// Only Microsoft issue warnings about correct behaviour...
#ifdef _MSVC_VER
#pragma warning(disable : 4351)
#endif

WgpuView3D::WgpuView3D(QCamera* cam, QRenderSettings* qrs, RenderSettings* rs, QWidget* parent)
  : QWidget(parent)
  , m_lastPos(0, 0)
  , m_initialized(false)
  , m_fakeHidden(false)
  , m_qrendersettings(qrs)
{
  m_viewerWindow = new ViewerWindow(rs);
  m_viewerWindow->gesture.input.setDoubleClickTime((double)QApplication::doubleClickInterval() / 1000.0);

  setAutoFillBackground(false);
  setAttribute(Qt::WA_PaintOnScreen);
  setAttribute(Qt::WA_DeleteOnClose);
  setFocusPolicy(Qt::StrongFocus);
  setMouseTracking(true);
  winId(); // create window handle

  m_qrendersettings->setRenderSettings(*rs);

  // IMPORTANT this is where the QT gui container classes send their values down into the
  // CScene object. GUI updates --> QT Object Changed() --> cam->Changed() -->
  // WgpuView3D->OnUpdateCamera
  QObject::connect(cam, SIGNAL(Changed()), this, SLOT(OnUpdateCamera()));
  QObject::connect(qrs, SIGNAL(Changed()), this, SLOT(OnUpdateQRenderSettings()));
  QObject::connect(qrs, SIGNAL(ChangedRenderer(int)), this, SLOT(OnUpdateRenderer(int)));

  // run a timer to update the clock
  // TODO is this different than using this->startTimer and QTimerEvent?
  m_etimer = new QTimer(parent);
  m_etimer->setTimerType(Qt::PreciseTimer);
  connect(m_etimer, &QTimer::timeout, this, [this] {
    // assume that in between QTimer events, true processEvents is called by Qt itself
    // QCoreApplication::processEvents();
    if (isEnabled()) {
      update();
    }
  });
  m_etimer->start();
}

void
WgpuView3D::initCameraFromImage(Scene* scene)
{
  // Tell the camera about the volume's bounding box
  m_viewerWindow->m_CCamera.m_SceneBoundingBox.m_MinP = scene->m_boundingBox.GetMinP();
  m_viewerWindow->m_CCamera.m_SceneBoundingBox.m_MaxP = scene->m_boundingBox.GetMaxP();
  // reposition to face image
  m_viewerWindow->m_CCamera.SetViewMode(ViewModeFront);

  RenderSettings* rs = m_viewerWindow->m_renderSettings;
  rs->m_DirtyFlags.SetFlag(CameraDirty);
}

void
WgpuView3D::toggleCameraProjection()
{
  ProjectionMode p = m_viewerWindow->m_CCamera.m_Projection;
  m_viewerWindow->m_CCamera.SetProjectionMode((p == PERSPECTIVE) ? ORTHOGRAPHIC : PERSPECTIVE);

  RenderSettings* rs = m_viewerWindow->m_renderSettings;
  rs->m_DirtyFlags.SetFlag(CameraDirty);
}

void
WgpuView3D::toggleAreaLightRotateControls()
{
  // toggle rotate tool
  if (m_areaLightMode == AREALIGHT_MODE::NONE || m_areaLightMode == AREALIGHT_MODE::TRANS) {
    m_viewerWindow->showAreaLightGizmo(true);
    m_viewerWindow->setTool(
      new RotateTool(m_viewerWindow->m_toolsUseLocalSpace, ManipulationTool::s_manipulatorSize * devicePixelRatioF()));
    m_viewerWindow->forEachTool(
      [this](ManipulationTool* tool) { tool->setUseLocalSpace(m_viewerWindow->m_toolsUseLocalSpace); });
    m_areaLightMode = AREALIGHT_MODE::ROT;
  } else {
    m_viewerWindow->showAreaLightGizmo(false);
    m_viewerWindow->setTool(nullptr);
    m_areaLightMode = AREALIGHT_MODE::NONE;
  }
}

void
WgpuView3D::onNewImage(Scene* scene)
{
  m_viewerWindow->m_renderer->setScene(scene);
  // costly teardown and rebuild.
  this->OnUpdateRenderer(m_viewerWindow->m_rendererType);
  // would be better to preserve renderer and just change the scene data to include the new image.
  // how tightly coupled is renderer and scene????
}

WgpuView3D::~WgpuView3D()
{
  wgpuSurfaceRelease(m_surface);
}

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
WgpuView3D::initializeGL(WGPUTextureView nextTexture)
{
  if (m_initialized) {
    return;
  }
  float dpr = devicePixelRatioF();

  LOG_INFO << "calling get_surface_from_canvas";

  m_surface = renderlib_wgpu::getSurfaceFromCanvas((void*)winId());

  WGPUAdapter adapter = renderlib_wgpu::getAdapter(m_surface);

  m_device = renderlib_wgpu::requestDevice(adapter);

  // set up swap chain
  m_swapChainFormat = WGPUTextureFormat_BGRA8Unorm; // wgpuSurfaceGetPreferredFormat(m_surface, adapter);
  m_surfaceConfig = {
    .nextInChain = NULL,
    .device = m_device,
    .format = m_swapChainFormat,
    .usage = WGPUTextureUsage_RenderAttachment,
    .viewFormatCount = 0,
    .viewFormats = NULL,
    .alphaMode = WGPUCompositeAlphaMode_Auto,
    .width = (uint32_t)(width() * dpr),
    .height = (uint32_t)(height() * dpr),
    .presentMode = WGPUPresentMode_Fifo,
  };
  wgpuSurfaceConfigure(m_surface, &m_surfaceConfig);

  // The WgpuView3D owns one CScene

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

  QSize newsize = size();
  m_viewerWindow->m_renderer->initialize(newsize.width() * dpr, newsize.height() * dpr);

  // Start timers
  m_etimer->start();

  // // Size viewport
  // resizeGL(newsize.width(), newsize.height());
}

void
WgpuView3D::paintEvent(QPaintEvent* e)
{
  if (!m_initialized) {
    return;
  }
  if (updatesEnabled()) {
    render();
    // the above render call should include this viewerwindow redraw
    //    m_viewerWindow->redraw();
  }
}

void
WgpuView3D::resizeGL(int w, int h)
{
  QResizeEvent e(QSize(w, h), QSize(w, h));
  resizeEvent(&e);
}
void
WgpuView3D::resizeEvent(QResizeEvent* event)
{
  if (event->size().isEmpty()) {
    m_fakeHidden = true;
    return;
  }
  m_fakeHidden = false;
  initializeGL(0);
  if (!m_initialized) {
    return;
  }

  float dpr = devicePixelRatioF();
  int w = event->size().width();
  int h = event->size().height();

  // (if w or h actually changed...)
  m_surfaceConfig.width = (uint32_t)(w * dpr);
  m_surfaceConfig.height = (uint32_t)(h * dpr);
  wgpuSurfaceConfigure(m_surface, &m_surfaceConfig);

  m_viewerWindow->setSize(w * dpr, h * dpr);
  m_viewerWindow->forEachTool(
    [this](ManipulationTool* tool) { tool->setSize(ManipulationTool::s_manipulatorSize * devicePixelRatioF()); });

  // update();
  //   invokeUserPaint();
}

static Gesture::Input::ButtonId
getButton(QMouseEvent* event)
{
  Gesture::Input::ButtonId btn;
  switch (event->button()) {
    case Qt::LeftButton:
      btn = Gesture::Input::ButtonId::kButtonLeft;
      break;
    case Qt::RightButton:
      btn = Gesture::Input::ButtonId::kButtonRight;
      break;
    case Qt::MiddleButton:
      btn = Gesture::Input::ButtonId::kButtonMiddle;
      break;
    default:
      btn = Gesture::Input::ButtonId::kButtonLeft;
      break;
  };
  return btn;
}
static int
getGestureMods(QMouseEvent* event)
{
  int mods = 0;
  if (event->modifiers() & Qt::ShiftModifier) {
    mods |= Gesture::Input::Mods::kShift;
  }
  if (event->modifiers() & Qt::ControlModifier) {
    mods |= Gesture::Input::Mods::kCtrl;
  }
  if (event->modifiers() & Qt::AltModifier) {
    mods |= Gesture::Input::Mods::kAlt;
  }
  if (event->modifiers() & Qt::MetaModifier) {
    mods |= Gesture::Input::Mods::kSuper;
  }
  return mods;
}

void
WgpuView3D::render()
{
  if (m_fakeHidden || !m_initialized) {
    return;
  }

  QWindow* win = windowHandle();
  if (!win || !win->isExposed()) {
    return;
  }
  WGPUSurfaceTexture nextTexture;

  wgpuSurfaceGetCurrentTexture(m_surface, &nextTexture);
  switch (nextTexture.status) {
    case WGPUSurfaceGetCurrentTextureStatus_Success:
      // All good, could check for `surface_texture.suboptimal` here.
      break;
    case WGPUSurfaceGetCurrentTextureStatus_Timeout:
    case WGPUSurfaceGetCurrentTextureStatus_Outdated:
    case WGPUSurfaceGetCurrentTextureStatus_Lost: {
      // Skip this frame, and re-configure surface.
      if (nextTexture.texture) {
        wgpuTextureRelease(nextTexture.texture);
      }
      if (width() != 0 && height() != 0) {
        m_surfaceConfig.width = (uint32_t)width();
        m_surfaceConfig.height = (uint32_t)height();
        wgpuSurfaceConfigure(m_surface, &m_surfaceConfig);
      }
      return;
    }
    case WGPUSurfaceGetCurrentTextureStatus_OutOfMemory:
    case WGPUSurfaceGetCurrentTextureStatus_DeviceLost:
    case WGPUSurfaceGetCurrentTextureStatus_Force32:
      // Fatal error
      LOG_ERROR << "get_current_texture status=" << nextTexture.status;
      abort();
  }
  assert(nextTexture.texture);

  WGPUTextureView frame = wgpuTextureCreateView(nextTexture.texture, NULL);
  assert(frame);

  renderWindowContents(frame);

  wgpuTextureViewRelease(frame);
  wgpuTextureRelease(nextTexture.texture);
}

void
WgpuView3D::renderWindowContents(WGPUTextureView nextTexture)
{
  if (!isEnabled()) {
    return;
  }

  WGPUCommandEncoderDescriptor commandEncoderDescriptor = { .label = "Command Encoder" };
  WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(m_device, &commandEncoderDescriptor);
  WGPURenderPassColorAttachment renderPassColorAttachment = {
    .view = nextTexture,
    .resolveTarget = nullptr,
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
    .nextInChain = NULL,
    .label = "Render Pass",
    .colorAttachmentCount = 1,
    .colorAttachments = &renderPassColorAttachment,
    .depthStencilAttachment = NULL,
    .occlusionQuerySet = 0,
    .timestampWrites = NULL,
  };
  WGPURenderPassEncoder renderPass = wgpuCommandEncoderBeginRenderPass(encoder, &renderPassDescriptor);
  wgpuRenderPassEncoderSetViewport(renderPass, 0, 0, width(), height(), 0, 1);
  // wgpuRenderPassEncoderSetPipeline(renderPass, pipeline);
  // wgpuRenderPassEncoderDraw(renderPass, 3, 1, 0, 0);
  wgpuRenderPassEncoderEnd(renderPass);

  WGPUQueue queue = wgpuDeviceGetQueue(m_device);
  WGPUCommandBufferDescriptor commandBufferDescriptor = { .label = NULL };
  WGPUCommandBuffer cmdBuffer = wgpuCommandEncoderFinish(encoder, &commandBufferDescriptor);
  wgpuQueueSubmit(queue, 1, &cmdBuffer);

  // wgpuCommandEncoderRelease(encoder);

  // TODO ENABLE THIS!!!
  // m_viewerWindow->redraw();

  wgpuSurfacePresent(m_surface);

  wgpuCommandBufferRelease(cmdBuffer);
  wgpuRenderPassEncoderRelease(renderPass);
  wgpuCommandEncoderRelease(encoder);
}

void
WgpuView3D::mousePressEvent(QMouseEvent* event)
{
  if (!isEnabled()) {
    return;
  }

  double time = Clock::now();
  const float dpr = devicePixelRatioF();
  m_viewerWindow->gesture.input.setButtonEvent(getButton(event),
                                               Gesture::Input::Action::kPress,
                                               getGestureMods(event),
                                               glm::vec2(event->x() * dpr, event->y() * dpr),
                                               time);
}

void
WgpuView3D::mouseReleaseEvent(QMouseEvent* event)
{
  if (!isEnabled()) {
    return;
  }

  double time = Clock::now();
  const float dpr = devicePixelRatioF();
  m_viewerWindow->gesture.input.setButtonEvent(getButton(event),
                                               Gesture::Input::Action::kRelease,
                                               getGestureMods(event),
                                               glm::vec2(event->x() * dpr, event->y() * dpr),
                                               time);
}

// No switch default to avoid -Wunreachable-code errors.
// However, this then makes -Wswitch-default complain.  Disable
// temporarily.
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-default"
#endif

void
WgpuView3D::mouseMoveEvent(QMouseEvent* event)
{
  if (!isEnabled()) {
    return;
  }
  const float dpr = devicePixelRatioF();

  m_viewerWindow->gesture.input.setPointerPosition(glm::vec2(event->x() * dpr, event->y() * dpr));
}

void
WgpuView3D::wheelEvent(QWheelEvent* event)
{
  if (!isEnabled()) {
    return;
  }
  const float dpr = devicePixelRatioF();

  // tell gesture there was a wheel event
  // m_viewerWindow->gesture.input.setPointerPosition(glm::vec2(event->x() * dpr, event->y() * dpr));
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

void
WgpuView3D::FitToScene()
{
  Scene* sc = m_viewerWindow->m_renderer->scene();

  glm::vec3 newPosition, newTarget;
  m_viewerWindow->m_CCamera.ComputeFitToBounds(sc->m_boundingBox, newPosition, newTarget);
  CameraAnimation anim = {};
  anim.duration = 0.5f; //< duration is seconds.
  anim.mod.position = newPosition - m_viewerWindow->m_CCamera.m_From;
  anim.mod.target = newTarget - m_viewerWindow->m_CCamera.m_Target;
  m_viewerWindow->m_cameraAnim.push_back(anim);
}

void
WgpuView3D::keyPressEvent(QKeyEvent* event)
{
  static enum MODE { NONE, ROT, TRANS } mode = MODE::NONE;

  if (event->key() == Qt::Key_A) {
    FitToScene();
  } else if (event->key() == Qt::Key_L) {
    // toggle showing area light gizmo
    m_viewerWindow->toggleAreaLightSelect();
  } else if (event->key() == Qt::Key_S) {
    // toggle local/global coordinates for transforms
    m_viewerWindow->m_toolsUseLocalSpace = !m_viewerWindow->m_toolsUseLocalSpace;
    m_viewerWindow->forEachTool(
      [this](ManipulationTool* tool) { tool->setUseLocalSpace(m_viewerWindow->m_toolsUseLocalSpace); });
  } else if (event->key() == Qt::Key_R) {
    // toggle rotate tool
    if (mode == MODE::NONE || mode == MODE::TRANS) {
      m_viewerWindow->setTool(new RotateTool(m_viewerWindow->m_toolsUseLocalSpace,
                                             ManipulationTool::s_manipulatorSize * devicePixelRatioF()));
      m_viewerWindow->forEachTool(
        [this](ManipulationTool* tool) { tool->setUseLocalSpace(m_viewerWindow->m_toolsUseLocalSpace); });
      mode = MODE::ROT;
    } else {
      m_viewerWindow->setTool(nullptr);
      mode = MODE::NONE;
    }
  } else if (event->key() == Qt::Key_T) {
    // toggle translate tool
    if (mode == MODE::NONE || mode == MODE::ROT) {
      m_viewerWindow->setTool(
        new MoveTool(m_viewerWindow->m_toolsUseLocalSpace, ManipulationTool::s_manipulatorSize * devicePixelRatioF()));
      m_viewerWindow->forEachTool(
        [this](ManipulationTool* tool) { tool->setUseLocalSpace(m_viewerWindow->m_toolsUseLocalSpace); });
      mode = MODE::TRANS;
    } else {
      m_viewerWindow->setTool(nullptr);
      mode = MODE::NONE;
    }
  } else {
    QWidget::keyPressEvent(event);
  }
}

void
WgpuView3D::OnUpdateCamera()
{
  //	QMutexLocker Locker(&gSceneMutex);
  RenderSettings* rs = m_viewerWindow->m_renderSettings;
  m_viewerWindow->m_CCamera.m_Film.m_Exposure = 1.0f - m_qcamera->GetFilm().GetExposure();
  m_viewerWindow->m_CCamera.m_Film.m_ExposureIterations = m_qcamera->GetFilm().GetExposureIterations();

  if (m_qcamera->GetFilm().IsDirty()) {
    const int FilmWidth = m_qcamera->GetFilm().GetWidth();
    const int FilmHeight = m_qcamera->GetFilm().GetHeight();

    m_viewerWindow->m_CCamera.m_Film.m_Resolution.SetResX(FilmWidth);
    m_viewerWindow->m_CCamera.m_Film.m_Resolution.SetResY(FilmHeight);
    m_viewerWindow->m_CCamera.Update();
    m_qcamera->GetFilm().UnDirty();

    rs->m_DirtyFlags.SetFlag(FilmResolutionDirty);
  }

  m_viewerWindow->m_CCamera.Update();

  // Aperture
  m_viewerWindow->m_CCamera.m_Aperture.m_Size = m_qcamera->GetAperture().GetSize();

  // Projection
  m_viewerWindow->m_CCamera.m_FovV = m_qcamera->GetProjection().GetFieldOfView();

  // Focus
  m_viewerWindow->m_CCamera.m_Focus.m_Type = (Focus::EType)m_qcamera->GetFocus().GetType();
  m_viewerWindow->m_CCamera.m_Focus.m_FocalDistance = m_qcamera->GetFocus().GetFocalDistance();

  rs->m_DenoiseParams.m_Enabled = m_qcamera->GetFilm().GetNoiseReduction();

  rs->m_DirtyFlags.SetFlag(CameraDirty);
}

void
WgpuView3D::OnUpdateQRenderSettings(void)
{
  // QMutexLocker Locker(&gSceneMutex);
  RenderSettings* rs = m_viewerWindow->m_renderSettings;

  rs->m_RenderSettings.m_DensityScale = m_qrendersettings->GetDensityScale();
  rs->m_RenderSettings.m_ShadingType = m_qrendersettings->GetShadingType();
  rs->m_RenderSettings.m_GradientFactor = m_qrendersettings->GetGradientFactor();

  // update window/levels / transfer function here!!!!

  rs->m_DirtyFlags.SetFlag(TransferFunctionDirty);
}

std::shared_ptr<CStatus>
WgpuView3D::getStatus()
{
  return m_viewerWindow->m_renderer->getStatusInterface();
}

void
WgpuView3D::OnUpdateRenderer(int rendererType)
{
  if (!isEnabled()) {
    LOG_ERROR << "attempted to update GLView3D renderer when view is disabled";
    return;
  }

  m_viewerWindow->setRenderer(rendererType);

  emit ChangedRenderer();
}

void
WgpuView3D::fromViewerState(const Serialize::ViewerState& s)
{
  m_qrendersettings->SetRendererType(s.rendererType == Serialize::RendererType_PID::PATHTRACE ? 1 : 0);

  // syntactic sugar
  CCamera& camera = m_viewerWindow->m_CCamera;

  camera.m_From = glm::vec3(s.camera.eye[0], s.camera.eye[1], s.camera.eye[2]);
  camera.m_Target = glm::vec3(s.camera.target[0], s.camera.target[1], s.camera.target[2]);
  camera.m_Up = glm::vec3(s.camera.up[0], s.camera.up[1], s.camera.up[2]);
  camera.m_FovV = s.camera.fovY;
  camera.SetProjectionMode(s.camera.projection == Serialize::Projection_PID::PERSPECTIVE ? PERSPECTIVE : ORTHOGRAPHIC);
  camera.m_OrthoScale = s.camera.orthoScale;

  camera.m_Film.m_Exposure = s.camera.exposure;
  camera.m_Aperture.m_Size = s.camera.aperture;
  camera.m_Focus.m_FocalDistance = s.camera.focalDistance;

  // TODO disentangle these QCamera* _camera and CCamera mCamera objects. Only CCamera should be necessary, I think.
  m_qcamera->GetProjection().SetFieldOfView(s.camera.fovY);
  m_qcamera->GetFilm().SetExposure(s.camera.exposure);
  m_qcamera->GetAperture().SetSize(s.camera.aperture);
  m_qcamera->GetFocus().SetFocalDistance(s.camera.focalDistance);
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
  if (!isEnabled()) {
    return QImage();
  }

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

  const float dpr = devicePixelRatioF();
  QOpenGLFramebufferObject* fbo =
    new QOpenGLFramebufferObject(width() * dpr, height() * dpr, fboFormat);
  check_gl("create fbo");

  fbo->bind();
  check_glfb("bind framebuffer for screen capture");

  // do a render into the temp framebuffer
  glViewport(0, 0, fbo->width(), fbo->height());
  m_viewerWindow->m_renderer->render(m_viewerWindow->m_CCamera);
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
  m_etimer->stop();
}

void
WgpuView3D::restartRenderLoop()
{
  m_etimer->start();
  std::shared_ptr<CStatus> s = getStatus();
  s->EnableUpdates(true);
}

WgpuCanvas::WgpuCanvas(QCamera* cam, QRenderSettings* qrs, RenderSettings* rs, QWidget* parent)
{
  setAttribute(Qt::WA_DeleteOnClose);
  setMouseTracking(true);

  m_view = new WgpuView3D(cam, qrs, rs, this);
  connect(m_view, SIGNAL(ChangedRenderer()), this, SLOT(OnChangedRenderer()));
  m_view->winId();

  m_layout = new QHBoxLayout(this);
  m_layout->setContentsMargins(0, 0, 0, 0);
  setLayout(m_layout);
  m_layout->addWidget(m_view);

  show();
}