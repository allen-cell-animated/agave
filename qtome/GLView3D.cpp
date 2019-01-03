#include "GLView3D.h"

#include "TransferFunction.h"
#include "ViewerState.h"

#include "renderlib/ImageXYZC.h"
#include "renderlib/Logging.h"
#include "renderlib/RenderGL.h"
#include "renderlib/RenderGLCuda.h"
#include "renderlib/RenderGLOptix.h"
#include "renderlib/RenderGLPT.h"
#include "renderlib/gl/Util.h"
#include "renderlib/gl/v33/V33Image3D.h"

#include <glm.h>

#include <QtGui/QMouseEvent>

#include <cmath>
#include <iostream>

// Only Microsoft issue warnings about correct behaviour...
#ifdef _MSVC_VER
#pragma warning(disable : 4351)
#endif

namespace {

void
qNormalizeAngle(int& angle)
{
  while (angle < 0)
    angle += 360 * 16;
  while (angle > 360 * 16)
    angle -= 360 * 16;
}

}

GLView3D::GLView3D(QCamera* cam, QTransferFunction* tran, RenderSettings* rs, QWidget* parent)
  : QOpenGLWidget(parent)
  , m_etimer()
  , m_lastPos(0, 0)
  , m_renderSettings(rs)
  , m_renderer(new RenderGLCuda(rs))
  ,
  //    _renderer(new RenderGL(img))
  m_qcamera(cam)
  , m_cameraController(cam, &m_CCamera)
  , m_transferFunction(tran)
  , m_rendererType(1)
{
  // The GLView3D owns one CScene

  m_cameraController.setRenderSettings(*m_renderSettings);
  m_transferFunction->setRenderSettings(*m_renderSettings);

  // IMPORTANT this is where the QT gui container classes send their values down into the CScene object.
  // GUI updates --> QT Object Changed() --> cam->Changed() --> GLView3D->OnUpdateCamera
  QObject::connect(cam, SIGNAL(Changed()), this, SLOT(OnUpdateCamera()));
  QObject::connect(tran, SIGNAL(Changed()), this, SLOT(OnUpdateTransferFunction()));
  QObject::connect(tran, SIGNAL(ChangedRenderer(int)), this, SLOT(OnUpdateRenderer(int)));
}

void
GLView3D::initCameraFromImage(Scene* scene)
{
  // Tell the camera about the volume's bounding box
  m_CCamera.m_SceneBoundingBox.m_MinP = scene->m_boundingBox.GetMinP();
  m_CCamera.m_SceneBoundingBox.m_MaxP = scene->m_boundingBox.GetMaxP();
  // reposition to face image
  m_CCamera.SetViewMode(ViewModeFront);
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
  m_renderer->initialize(newsize.width(), newsize.height());

  // Start timers
  startTimer(0);
  m_etimer.start();

  // Size viewport
  resizeGL(newsize.width(), newsize.height());
}

void
GLView3D::paintGL()
{
  makeCurrent();

  m_CCamera.Update();

  m_renderer->render(m_CCamera);
}

void
GLView3D::resizeGL(int w, int h)
{
  makeCurrent();

  m_CCamera.m_Film.m_Resolution.SetResX(w);
  m_CCamera.m_Film.m_Resolution.SetResY(h);
  m_renderer->resize(w, h);
}

void
GLView3D::mousePressEvent(QMouseEvent* event)
{
  m_lastPos = event->pos();
  m_cameraController.m_OldPos[0] = m_lastPos.x();
  m_cameraController.m_OldPos[1] = m_lastPos.y();
}

void
GLView3D::mouseReleaseEvent(QMouseEvent* event)
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
  m_cameraController.OnMouseMove(event);
  m_lastPos = event->pos();
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

void
GLView3D::timerEvent(QTimerEvent* event)
{
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
    // 		//
    rs.m_DirtyFlags.SetFlag(FilmResolutionDirty);
  }

  m_CCamera.Update();

  // Aperture
  m_CCamera.m_Aperture.m_Size = m_qcamera->GetAperture().GetSize();

  // Projection
  m_CCamera.m_FovV = m_qcamera->GetProjection().GetFieldOfView();

  // Focus
  m_CCamera.m_Focus.m_Type = (CFocus::EType)m_qcamera->GetFocus().GetType();
  m_CCamera.m_Focus.m_FocalDistance = m_qcamera->GetFocus().GetFocalDistance();

  rs.m_DenoiseParams.m_Enabled = m_qcamera->GetFilm().GetNoiseReduction();

  rs.m_DirtyFlags.SetFlag(CameraDirty);
}
void
GLView3D::OnUpdateTransferFunction(void)
{
  // QMutexLocker Locker(&gSceneMutex);
  RenderSettings& rs = *m_renderSettings;

  rs.m_RenderSettings.m_DensityScale = m_transferFunction->GetDensityScale();
  rs.m_RenderSettings.m_ShadingType = m_transferFunction->GetShadingType();
  rs.m_RenderSettings.m_GradientFactor = m_transferFunction->GetGradientFactor();

  // update window/levels / transfer function here!!!!

  rs.m_DirtyFlags.SetFlag(TransferFunctionDirty);
}

CStatus*
GLView3D::getStatus()
{
  return m_renderer->getStatusInterface();
}

void
GLView3D::OnUpdateRenderer(int rendererType)
{
  makeCurrent();

  // clean up old renderer.
  if (m_renderer) {
    m_renderer->cleanUpResources();
  }

  Scene* sc = m_renderer->scene();

  switch (rendererType) {
    case 1:
      LOG_DEBUG << "Set CUDA Renderer";
      m_renderer.reset(new RenderGLCuda(m_renderSettings));
      m_renderSettings->m_DirtyFlags.SetFlag(TransferFunctionDirty);
      break;
    case 2:
      LOG_DEBUG << "Set OpenGL pathtrace Renderer";
      m_renderer.reset(new RenderGLPT(m_renderSettings));
      m_renderSettings->m_DirtyFlags.SetFlag(TransferFunctionDirty);
      break;
    case 3:
      LOG_DEBUG << "Set OptiX Renderer";
      m_renderer.reset(new RenderGLOptix(m_renderSettings));
      m_renderSettings->m_DirtyFlags.SetFlag(MeshDirty);
      break;
    default:
      LOG_DEBUG << "Set OpenGL Renderer";
      m_renderer.reset(new RenderGL(m_renderSettings));
  };
  m_rendererType = rendererType;

  QSize newsize = size();
  // need to update the scene in QAppearanceSettingsWidget.
  m_renderer->setScene(sc);
  m_renderer->initialize(newsize.width(), newsize.height());

  m_renderSettings->m_DirtyFlags.SetFlag(RenderParamsDirty);

  emit ChangedRenderer();
}

void
GLView3D::fromViewerState(const ViewerState& s)
{
  m_CCamera.m_From = glm::vec3(s.m_eyeX, s.m_eyeY, s.m_eyeZ);
  m_CCamera.m_Target = glm::vec3(s.m_targetX, s.m_targetY, s.m_targetZ);
  m_CCamera.m_Up = glm::vec3(s.m_upX, s.m_upY, s.m_upZ);
  m_CCamera.m_FovV = s.m_fov;

  m_CCamera.m_Film.m_Exposure = s.m_exposure;
  m_CCamera.m_Aperture.m_Size = s.m_apertureSize;
  m_CCamera.m_Focus.m_FocalDistance = s.m_focalDistance;

  // TODO disentangle these QCamera* _camera and CCamera mCamera objects. Only CCamera should be necessary, I think.
  m_qcamera->GetProjection().SetFieldOfView(s.m_fov);
  m_qcamera->GetFilm().SetExposure(s.m_exposure);
  m_qcamera->GetAperture().SetSize(s.m_apertureSize);
  m_qcamera->GetFocus().SetFocalDistance(s.m_focalDistance);
}
