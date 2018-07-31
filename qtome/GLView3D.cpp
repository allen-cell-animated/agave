#include "GLView3D.h"

#include "TransferFunction.h"

#include "renderlib/gl/v33/V33Image3D.h"
#include "renderlib/ImageXYZC.h"
#include "renderlib/Logging.h"
#include "renderlib/RenderGL.h"
#include "renderlib/RenderGLCuda.h"
#include "renderlib/RenderGLOptix.h"
#include <glm.h>

#include <QtGui/QMouseEvent>

#include <cmath>
#include <iostream>

// Only Microsoft issue warnings about correct behaviour...
#ifdef _MSVC_VER
#pragma warning(disable : 4351)
#endif

namespace
{

  void
  qNormalizeAngle(int &angle)
  {
    while (angle < 0)
      angle += 360 * 16;
    while (angle > 360 * 16)
      angle -= 360 * 16;
  }

}

GLView3D::GLView3D(QCamera* cam,
	QTransferFunction* tran,
	RenderSettings* rs,
    QWidget* /* parent */):
    GLWindow(),
    etimer(),
    lastPos(0, 0),
	_renderSettings(rs),
    _renderer(new RenderGLCuda(rs)),
	//    _renderer(new RenderGL(img))
	_camera(cam),
	_cameraController(cam, &mCamera),
	_transferFunction(tran),
	_rendererType(1)
{
	// The GLView3D owns one CScene

	_cameraController.setRenderSettings(*_renderSettings);
	_transferFunction->setRenderSettings(*_renderSettings);

	// IMPORTANT this is where the QT gui container classes send their values down into the CScene object.
	// GUI updates --> QT Object Changed() --> cam->Changed() --> GLView3D->OnUpdateCamera
	QObject::connect(cam, SIGNAL(Changed()), this, SLOT(OnUpdateCamera()));
	QObject::connect(tran, SIGNAL(Changed()), this, SLOT(OnUpdateTransferFunction()));
	QObject::connect(tran, SIGNAL(ChangedRenderer(int)), this, SLOT(OnUpdateRenderer(int)));
}

void GLView3D::onNewImage(Scene* scene)
{
	// Tell the camera about the volume's bounding box
	mCamera.m_SceneBoundingBox.m_MinP = scene->_boundingBox.GetMinP();
	mCamera.m_SceneBoundingBox.m_MaxP = scene->_boundingBox.GetMaxP();
	// reposition to face image
	mCamera.SetViewMode(ViewModeFront);

	_renderer->setScene(scene);
	// costly teardown and rebuild.
	this->OnUpdateRenderer(_rendererType);
	// would be better to preserve renderer and just change the scene data to include the new image.
	// how tightly coupled is renderer and scene????
}

GLView3D::~GLView3D()
{
    makeCurrent();
}

QSize GLView3D::minimumSizeHint() const
{
    return QSize(800, 600);
}

QSize GLView3D::sizeHint() const
{
    return QSize(800, 600);
}

void
GLView3D::initialize()
{
    makeCurrent();

    QSize newsize = size();
    _renderer->initialize(newsize.width(), newsize.height());

    // Start timers
    startTimer(0);
    etimer.start();

    // Size viewport
    resize();
}

void
GLView3D::render()
{
    makeCurrent();

	mCamera.Update();
    
	_renderer->render(mCamera);
}

void
GLView3D::resize()
{
    makeCurrent();

    QSize newsize = size();
	mCamera.m_Film.m_Resolution.SetResX(newsize.width());
	mCamera.m_Film.m_Resolution.SetResY(newsize.height());
	_renderer->resize(newsize.width(), newsize.height());
}


void
GLView3D::mousePressEvent(QMouseEvent *event)
{
    lastPos = event->pos();
    _cameraController.m_OldPos[0] = lastPos.x();
    _cameraController.m_OldPos[1] = lastPos.y();
}
    
void
GLView3D::mouseReleaseEvent(QMouseEvent *event)
{
    lastPos = event->pos();
    _cameraController.m_OldPos[0] = lastPos.x();
    _cameraController.m_OldPos[1] = lastPos.y();
}

// No switch default to avoid -Wunreachable-code errors.
// However, this then makes -Wswitch-default complain.  Disable
// temporarily.
#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wswitch-default"
#endif

// x, y in 0..1 relative to screen
glm::vec3 get_arcball_vector(float xndc, float yndc) {
    glm::vec3 P = glm::vec3(1.0*xndc * 2 - 1.0,
    1.0*yndc * 2 - 1.0,
    0);
    P.y = -P.y;
    float OP_squared = P.x * P.x + P.y * P.y;
    if (OP_squared <= 1 * 1)
    P.z = sqrt(1 * 1 - OP_squared);  // Pythagore
    else
    P = glm::normalize(P);  // nearest point
    return P;
}

void
GLView3D::mouseMoveEvent(QMouseEvent *event)
{
    _cameraController.OnMouseMove(event);
    lastPos = event->pos();
}

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif

void
GLView3D::timerEvent (QTimerEvent *event)
{
    makeCurrent();

    GLWindow::timerEvent(event);

    renderLater();
}


void GLView3D::OnUpdateCamera()
{
	//	QMutexLocker Locker(&gSceneMutex);
	RenderSettings& rs = *_renderSettings;
	mCamera.m_Film.m_Exposure = 1.0f - _camera->GetFilm().GetExposure();
	mCamera.m_Film.m_ExposureIterations = _camera->GetFilm().GetExposureIterations();

	if (_camera->GetFilm().IsDirty())
	{
		const int FilmWidth = _camera->GetFilm().GetWidth();
		const int FilmHeight = _camera->GetFilm().GetHeight();

		mCamera.m_Film.m_Resolution.SetResX(FilmWidth);
		mCamera.m_Film.m_Resolution.SetResY(FilmHeight);
		mCamera.Update();
		_camera->GetFilm().UnDirty();
		// 		// 
		rs.m_DirtyFlags.SetFlag(FilmResolutionDirty);
	}

	mCamera.Update();

	// Aperture
	mCamera.m_Aperture.m_Size = _camera->GetAperture().GetSize();

	// Projection
	mCamera.m_FovV = _camera->GetProjection().GetFieldOfView();

	// Focus
	mCamera.m_Focus.m_Type = (CFocus::EType)_camera->GetFocus().GetType();
	mCamera.m_Focus.m_FocalDistance = _camera->GetFocus().GetFocalDistance();

	rs.m_DenoiseParams.m_Enabled = _camera->GetFilm().GetNoiseReduction();

	rs.m_DirtyFlags.SetFlag(CameraDirty);
}
void GLView3D::OnUpdateTransferFunction(void)
{
	//QMutexLocker Locker(&gSceneMutex);
	RenderSettings& rs = *_renderSettings;

	rs.m_RenderSettings.m_DensityScale = _transferFunction->GetDensityScale();
	rs.m_RenderSettings.m_ShadingType = _transferFunction->GetShadingType();
	rs.m_RenderSettings.m_GradientFactor = _transferFunction->GetGradientFactor();

	// update window/levels / transfer function here!!!!

	rs.m_DirtyFlags.SetFlag(TransferFunctionDirty);
}

CStatus* GLView3D::getStatus() {
	return _renderer->getStatusInterface();
}

void GLView3D::OnUpdateRenderer(int rendererType)
{
	makeCurrent();

	// clean up old renderer.
	if (_renderer) {
		_renderer->cleanUpResources();
	}


	Scene* sc = _renderer->scene();
	
	switch (rendererType) {
	case 1:
		LOG_DEBUG << "Set CUDA Renderer";
		_renderer.reset(new RenderGLCuda(_renderSettings));
		break;
	case 2:
		LOG_DEBUG << "Set OptiX Renderer";
		_renderer.reset(new RenderGLOptix(_renderSettings));
		break;
	default:
		LOG_DEBUG << "Set OpenGL Renderer";
		_renderer.reset(new RenderGL(_renderSettings));
	};
	_rendererType = rendererType;

	QSize newsize = size();
	// need to update the scene in QAppearanceSettingsWidget.
	_renderer->setScene(sc);
	_renderer->initialize(newsize.width(), newsize.height());

	_renderSettings->m_DirtyFlags.SetFlag(RenderParamsDirty);
}
