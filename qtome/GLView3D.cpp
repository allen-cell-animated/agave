#include "GLView3D.h"

#include "TransferFunction.h"

#include "renderlib/gl/v33/V33Image3D.h"
#include "renderlib/RenderGL.h"
#include "renderlib/RenderGLCuda.h"
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

GLView3D::GLView3D(std::shared_ptr<ImageXYZC>  img,
	QCamera* cam,
	QTransferFunction* tran,
	CScene* scene,
    QWidget* /* parent */):
    GLWindow(),
    camera(),
    mouseMode(MODE_ROTATE),
    etimer(),
    cmin(0.0f),
    cmax(1.0f),
    plane(0), _z(0), _c(0),
    oldplane(-1),
    lastPos(0, 0),
    _img(img),
    _renderGL(new RenderGLCuda(img, scene)),
	//    _renderGL(new RenderGL(img))
	_camera(cam),
	_cameraController(cam),
	_transferFunction(tran)
{
	// The GLView3D owns one CScene

	_cameraController.setScene(_renderGL->getScene());
	_transferFunction->setScene(_renderGL->getScene());

	// IMPORTANT this is where the QT gui container classes send their values down into the CScene object.
	// GUI updates --> QT Object Changed() --> cam->Changed() --> GLView3D->OnUpdateCamera
	QObject::connect(cam, SIGNAL(Changed()), this, SLOT(OnUpdateCamera()));
	QObject::connect(tran, SIGNAL(Changed()), this, SLOT(OnUpdateTransferFunction()));

	camera.position = glm::vec3(0.0, 0.0, 2.5);
	camera.up = glm::vec3(0.0, 1.0, 0.0);
	glm::vec3 target(0.0, 0.0, 0.0); // position of model or world center!!!
	camera.direction = glm::normalize(target - camera.position);
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

int
GLView3D::getZoom() const
{
    return camera.zoom;
}

int
GLView3D::getXTranslation() const
{
    return camera.xTran;
}

int
GLView3D::getYTranslation() const
{
    return camera.yTran;
}

int
GLView3D::getZRotation() const
{
    return camera.zRot;
}

int
GLView3D::getChannelMin() const
{
    return static_cast<int>(cmin[0] * 255.0*16.0);
}

int
GLView3D::getChannelMax() const
{
    return static_cast<int>(cmax[0] * 255.0*16.0);
}

size_t
GLView3D::getPlane() const
{
    return plane;
}
	
size_t GLView3D::getZ() const { return _z; }
size_t GLView3D::getC() const { return _c; }

void
GLView3D::setZoom(int zoom)
{
    if (zoom != camera.zoom) {
    camera.zoom = zoom;
    emit zoomChanged(zoom);
    renderLater();
    }
}

void
GLView3D::setXTranslation(int xtran)
{
    if (xtran != camera.xTran) {
    camera.xTran = xtran;
    emit xTranslationChanged(xtran);
    renderLater();
    }
}

void
GLView3D::setYTranslation(int ytran)
{
    if (ytran != camera.yTran) {
    camera.yTran = ytran;
    emit yTranslationChanged(ytran);
    renderLater();
    }
}

void
GLView3D::setZRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != camera.zRot) {
    camera.zRot = angle;
    emit zRotationChanged(angle);
    renderLater();
    }
}

void
GLView3D::setMouseMode(MouseMode mode)
{
    mouseMode = mode;
}

GLView3D::MouseMode
GLView3D::getMouseMode() const
{
    return mouseMode;
}


// Note fixed to one channel at the moment.

void GLView3D::setChannelMin(int min)
{
    float v = min / (255.0*16.0);
    if (cmin[0] != v)
    {
        cmin = glm::vec3(v);
        emit channelMinChanged(min);
        renderLater();
    }
    if (cmin[0] > cmax[0])
    setChannelMax(min);
}

void
GLView3D::setChannelMax(int max)
{
    float v = max / (255.0*16.0);
    if (cmax[0] != v)
    {
        cmax = glm::vec3(v);
        emit channelMaxChanged(max);
        renderLater();
    }
    if (cmax[0] < cmin[0])
    setChannelMin(max);
}

void
GLView3D::setZCPlane(size_t z, size_t c)
{
    if (_z != z || _c != c) {
    _z = z;
    _c = c;
	_renderGL->setChannel((int)c);
    renderLater();
    }
}

void
GLView3D::setPlane(size_t plane)
{
    if (this->plane != plane)
    {
        this->plane = plane;
        emit planeChanged(plane);
        renderLater();
    }
}

void
GLView3D::initialize()
{
    makeCurrent();

    QSize newsize = size();
    _renderGL->initialize(newsize.width(), newsize.height());

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

    _renderGL->render(camera);
}

void
GLView3D::resize()
{
    makeCurrent();

    QSize newsize = size();
    _renderGL->resize(newsize.width(), newsize.height());
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


    QSize s = size();
    int dx = event->x() - lastPos.x();
    int dy = event->y() - lastPos.y();

    glm::vec3 right = glm::cross(camera.direction, camera.up);
    if (event->buttons() & Qt::LeftButton) {
    switch (mouseMode)
        {
        case MODE_ZOOM:
        setZoom(camera.zoom + 8 * dy);
        break;
        case MODE_PAN:
        camera.translate(right*(-dx*0.002f) + camera.up*(dy*0.002f));
        break;
        case MODE_ROTATE:
        float lastxndc = float(lastPos.x()) / float(s.width());
        float lastyndc = float(lastPos.y()) / float(s.height());
        glm::vec3 va = get_arcball_vector(lastxndc, lastyndc);
        float xndc = float(event->x()) / float(s.width());
        float yndc = float(event->y()) / float(s.height());
        glm::vec3 vb = get_arcball_vector(xndc, yndc);

        float angle = 0.02 * acos(min(1.0f, glm::dot(va, vb)));
        glm::vec3 axis_in_camera_coord = glm::cross(va, vb);

        // just tumble the world model matrix, for now...
        glm::mat3 camera2object = glm::inverse(glm::mat3(camera.view) * glm::mat3(camera.model));
        glm::vec3 axis_in_object_coord = camera2object * axis_in_camera_coord;
        camera.model = glm::rotate(camera.model, glm::degrees(angle), axis_in_object_coord);
        break;
        }
    }
    lastPos = event->pos();
}

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif

void
GLView3D::timerEvent (QTimerEvent *event)
{
    makeCurrent();

	// Window size.  Size may be zero if the window is not yet mapped.
	QSize s = size();

	camera.projectionType = Camera::PERSPECTIVE;
	if (camera.projectionType == Camera::PERSPECTIVE) {
		camera.update();

		camera.projection = glm::perspective(glm::radians(45.0f), static_cast<float>(s.width()) / static_cast<float>(s.height()), 0.01f, 10.f);
	}
	else {
		float zoomfactor = camera.zoomfactor();

		float xtr(static_cast<float>(camera.xTran) / zoomfactor);
		float ytr(static_cast<float>(camera.yTran) / zoomfactor);

		glm::vec3 tr(glm::rotateZ(glm::vec3(xtr, ytr, 0.0), camera.rotation()));

		camera.view = glm::lookAt(glm::vec3(tr[0], tr[1], 5.0),
			glm::vec3(tr[0], tr[1], 0.0),
			glm::rotateZ(glm::vec3(0.0, 1.0, 0.0), camera.rotation()));

		float xrange = static_cast<float>(s.width()) / zoomfactor;
		float yrange = static_cast<float>(s.height()) / zoomfactor;

		camera.projection = glm::ortho(-xrange, xrange,
			-yrange, yrange,
			0.0f, 10.0f);
	}

	if (_renderGL->getImage()) {
		_renderGL->getImage()->setPlane((int)getPlane(), (int)getZ(), (int)getC());
		_renderGL->getImage()->setMin(cmin);
		_renderGL->getImage()->setMax(cmax);
	}

    GLWindow::timerEvent(event);

    renderLater();
}


void GLView3D::OnUpdateCamera()
{
	//	QMutexLocker Locker(&gSceneMutex);
	CScene& scene = _renderGL->getScene();
	scene.m_Camera.m_Film.m_Exposure = 1.0f - _camera->GetFilm().GetExposure();
	scene.m_Camera.m_Film.m_ExposureIterations = _camera->GetFilm().GetExposureIterations();

	if (_camera->GetFilm().IsDirty())
	{
		const int FilmWidth = _camera->GetFilm().GetWidth();
		const int FilmHeight = _camera->GetFilm().GetHeight();

		scene.m_Camera.m_Film.m_Resolution.SetResX(FilmWidth);
		scene.m_Camera.m_Film.m_Resolution.SetResY(FilmHeight);
		scene.m_Camera.Update();
		_camera->GetFilm().UnDirty();
		// 		// 
		scene.m_DirtyFlags.SetFlag(FilmResolutionDirty);
	}

	// 	gScene.m_Camera.m_From	= gCamera.GetFrom();
	// 	gScene.m_Camera.m_Target	= gCamera.GetTarget();
	// 	gScene.m_Camera.m_Up		= gCamera.GetUp();

	scene.m_Camera.Update();

	// Aperture
	scene.m_Camera.m_Aperture.m_Size = _camera->GetAperture().GetSize();

	// Projection
	scene.m_Camera.m_FovV = _camera->GetProjection().GetFieldOfView();

	// Focus
	scene.m_Camera.m_Focus.m_Type = (CFocus::EType)_camera->GetFocus().GetType();
	scene.m_Camera.m_Focus.m_FocalDistance = _camera->GetFocus().GetFocalDistance();

	scene.m_DenoiseParams.m_Enabled = _camera->GetFilm().GetNoiseReduction();

	scene.m_DirtyFlags.SetFlag(CameraDirty);
}
void GLView3D::OnUpdateTransferFunction(void)
{
	//QMutexLocker Locker(&gSceneMutex);
	CScene& scene = _renderGL->getScene();

	scene.m_DensityScale = _transferFunction->GetDensityScale();
	scene.m_ShadingType = _transferFunction->GetShadingType();
	scene.m_GradientFactor = _transferFunction->GetGradientFactor();

	scene.m_DirtyFlags.SetFlag(TransferFunctionDirty);
}
