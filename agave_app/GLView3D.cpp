#include "GLView3D.h"

#include "QRenderSettings.h"
#include "ViewerState.h"

#include "renderlib/ImageXYZC.h"
#include "renderlib/Logging.h"
#include "renderlib/RenderGL.h"
#include "renderlib/RenderGLPT.h"
#include "renderlib/gl/Image3D.h"
#include "renderlib/gl/Util.h"

#include <glm.h>

#include <QGuiApplication>
#include <QMouseEvent>
#include <QOpenGLFramebufferObject>
#include <QOpenGLFramebufferObjectFormat>
#include <QScreen>
#include <QWindow>

#include <cmath>
#include <iostream>

// Only Microsoft issue warnings about correct behaviour...
#ifdef _MSVC_VER
#pragma warning(disable : 4351)
#endif

GLView3D::GLView3D(QCamera* cam, QRenderSettings* qrs, RenderSettings* rs, QWidget* parent)
  : QOpenGLWidget(parent)
  , m_etimer()
  , m_lastPos(0, 0)
  , m_renderSettings(rs)
  , m_renderer(new RenderGLPT(rs))
  ,
  //    _renderer(new RenderGL(img))
  m_qcamera(cam)
  , m_cameraController(cam, &m_CCamera)
  , m_qrendersettings(qrs)
  , m_rendererType(1)
{
  // The GLView3D owns one CScene

  m_cameraController.setRenderSettings(*m_renderSettings);
  m_qrendersettings->setRenderSettings(*m_renderSettings);

  // IMPORTANT this is where the QT gui container classes send their values down into the CScene object.
  // GUI updates --> QT Object Changed() --> cam->Changed() --> GLView3D->OnUpdateCamera
  QObject::connect(cam, SIGNAL(Changed()), this, SLOT(OnUpdateCamera()));
  QObject::connect(qrs, SIGNAL(Changed()), this, SLOT(OnUpdateQRenderSettings()));
  QObject::connect(qrs, SIGNAL(ChangedRenderer(int)), this, SLOT(OnUpdateRenderer(int)));
}

void
GLView3D::initCameraFromImage(Scene* scene)
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
GLView3D::toggleCameraProjection()
{
  ProjectionMode p = m_CCamera.m_Projection;
  m_CCamera.SetProjectionMode((p == PERSPECTIVE) ? ORTHOGRAPHIC : PERSPECTIVE);

  RenderSettings& rs = *m_renderSettings;
  rs.m_DirtyFlags.SetFlag(CameraDirty);
}

void
GLView3D::onNewImage(Scene* scene)
{
  m_renderer->setScene(scene);
  // costly teardown and rebuild.
  this->OnUpdateRenderer(m_rendererType);
  // would be better to preserve renderer and just change the scene data to include the new image.
  // how tightly coupled is renderer and scene????
}

GLView3D::~GLView3D()
{
  makeCurrent();
  check_gl("view dtor makecurrent");
  // doneCurrent();
}

QSize
GLView3D::minimumSizeHint() const
{
  return QSize(800, 600);
}

QSize
GLView3D::sizeHint() const
{
  return QSize(800, 600);
}

void
GLView3D::initializeGL()
{

  makeCurrent();

  QSize newsize = size();
  m_renderer->initialize(newsize.width(), newsize.height(), devicePixelRatioF());

  // Start timers
  startTimer(0);
  m_etimer.start();

  // Size viewport
  resizeGL(newsize.width(), newsize.height());
}

void
GLView3D::paintGL()
{
  if (!isEnabled()) {
    return;
  }
  makeCurrent();

  m_CCamera.Update();

  m_renderer->render(m_CCamera);

  doneCurrent();
}

void
GLView3D::resizeGL(int w, int h)
{
  if (!isEnabled()) {
    return;
  }
  makeCurrent();

  m_CCamera.m_Film.m_Resolution.SetResX(w);
  m_CCamera.m_Film.m_Resolution.SetResY(h);
  m_renderer->resize(w, h, devicePixelRatioF());

  doneCurrent();
}

void
GLView3D::mousePressEvent(QMouseEvent* event)
{
  if (!isEnabled()) {
    return;
  }
  m_lastPos = event->pos();
  m_cameraController.m_OldPos[0] = m_lastPos.x();
  m_cameraController.m_OldPos[1] = m_lastPos.y();
}

void
GLView3D::mouseReleaseEvent(QMouseEvent* event)
{
  if (!isEnabled()) {
    return;
  }
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
glm::vec3
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
GLView3D::mouseMoveEvent(QMouseEvent* event)
{
  if (!isEnabled()) {
    return;
  }
  m_cameraController.OnMouseMove(event);
  m_lastPos = event->pos();
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

void
GLView3D::timerEvent(QTimerEvent* event)
{
  if (!isEnabled()) {
    return;
  }

  makeCurrent();

  QOpenGLWidget::timerEvent(event);

  update();
}

void
GLView3D::OnUpdateCamera()
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
GLView3D::OnUpdateQRenderSettings(void)
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
GLView3D::getStatus()
{
  return m_renderer->getStatusInterface();
}

void
GLView3D::OnUpdateRenderer(int rendererType)
{
  if (!isEnabled()) {
    LOG_ERROR << "attempted to update GLView3D renderer when view is disabled";
    return;
  }

  makeCurrent();

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
}

void
GLView3D::fromViewerState(const Serialize::ViewerState& s)
{
  m_CCamera.m_From = glm::make_vec3(s.camera.eye.data());
  m_CCamera.m_Target = glm::make_vec3(s.camera.target.data());
  m_CCamera.m_Up = glm::make_vec3(s.camera.up.data());
  m_CCamera.m_FovV = s.camera.fovY;
  m_CCamera.SetProjectionMode(s.camera.projection == Serialize::Projection_PID::PERSPECTIVE ? PERSPECTIVE
                                                                                            : ORTHOGRAPHIC);
  m_CCamera.m_OrthoScale = s.camera.orthoScale;

  m_CCamera.m_Film.m_Exposure = s.camera.exposure;
  m_CCamera.m_Aperture.m_Size = s.camera.aperture;
  m_CCamera.m_Focus.m_FocalDistance = s.camera.focalDistance;

  // TODO disentangle these QCamera* _camera and CCamera mCamera objects. Only CCamera should be necessary, I think.
  m_qcamera->GetProjection().SetFieldOfView(s.camera.fovY);
  m_qcamera->GetFilm().SetExposure(s.camera.exposure);
  m_qcamera->GetAperture().SetSize(s.camera.aperture);
  m_qcamera->GetFocus().SetFocalDistance(s.camera.focalDistance);
}

QPixmap
GLView3D::capture()
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
GLView3D::captureQimage()
{
  if (!isEnabled()) {
    return QImage();
  }

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
}

void
GLView3D::pauseRenderLoop()
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
GLView3D::restartRenderLoop()
{
  m_etimer.restart();
  std::shared_ptr<CStatus> s = getStatus();
  s->EnableUpdates(true);
}
