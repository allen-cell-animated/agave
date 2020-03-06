#include "GLView2D.h"

#include "renderlib/gl/Image2D.h"
#include "renderlib/RenderGL2d.h"
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


GLView2D::GLView2D(std::shared_ptr<ImageXYZC>  img,
                    QWidget                                                * /* parent */):
    GLWindow(),
    camera(),
    mouseMode(MODE_ZOOM),
    etimer(),
    cmin(0.0f),
    cmax(1.0f),
    plane(0), _z(0), _c(0),
    oldplane(-1),
    lastPos(0, 0),
	_img(img),
	_renderGL(new RenderGL2d(img))
{
}

GLView2D::~GLView2D()
{
    makeCurrent();
}

QSize GLView2D::minimumSizeHint() const
{
    return QSize(800, 600);
}

QSize GLView2D::sizeHint() const
{
    return QSize(800, 600);
}

int
GLView2D::getZoom() const
{
    return camera.zoom;
}

int
GLView2D::getXTranslation() const
{
    return camera.xTran;
}

int
GLView2D::getYTranslation() const
{
    return camera.yTran;
}

int
GLView2D::getZRotation() const
{
    return camera.zRot;
}

int
GLView2D::getChannelMin() const
{
    return static_cast<int>(cmin[0] * 255.0*16.0);
}

int
GLView2D::getChannelMax() const
{
    return static_cast<int>(cmax[0] * 255.0*16.0);
}

size_t
GLView2D::getPlane() const
{
    return plane;
}
size_t GLView2D::getZ() const { return _z; }
size_t GLView2D::getC() const { return _c; }

void
GLView2D::setZoom(int zoom)
{
    if (zoom != camera.zoom) {
    camera.zoom = zoom;
    emit zoomChanged(zoom);
    renderLater();
    }
}

void
GLView2D::setXTranslation(int xtran)
{
    if (xtran != camera.xTran) {
    camera.xTran = xtran;
    emit xTranslationChanged(xtran);
    renderLater();
    }
}

void
GLView2D::setYTranslation(int ytran)
{
    if (ytran != camera.yTran) {
    camera.yTran = ytran;
    emit yTranslationChanged(ytran);
    renderLater();
    }
}

void
GLView2D::setZRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != camera.zRot) {
    camera.zRot = angle;
    emit zRotationChanged(angle);
    renderLater();
    }
}

void
GLView2D::setMouseMode(MouseMode mode)
{
    mouseMode = mode;
}

GLView2D::MouseMode
GLView2D::getMouseMode() const
{
    return mouseMode;
}


// Note fixed to one channel at the moment.

void GLView2D::setChannelMin(int min)
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
GLView2D::setChannelMax(int max)
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
	GLView2D::setZCPlane(size_t z, size_t c)
{
	if (_z != z || _c != c) {
		_z = z;
		_c = c;
		renderLater();
	}
}

void
GLView2D::setPlane(size_t plane)
{
    if (this->plane != plane)
    {
        this->plane = plane;
        emit planeChanged(plane);
        renderLater();
    }
}

void
GLView2D::initialize()
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
GLView2D::render()
{
    makeCurrent();

	_renderGL->render(camera);
}

void
GLView2D::resize()
{
    makeCurrent();

    QSize newsize = size();
	_renderGL->resize(newsize.width(), newsize.height());
}


void
GLView2D::mousePressEvent(QMouseEvent *event)
{
    lastPos = event->pos();
}

// No switch default to avoid -Wunreachable-code errors.
// However, this then makes -Wswitch-default complain.  Disable
// temporarily.
#ifdef __GNUC__
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wswitch-default"
#endif

void
GLView2D::mouseMoveEvent(QMouseEvent *event)
{
    int dx = event->x() - lastPos.x();
    int dy = event->y() - lastPos.y();

    if (event->buttons() & Qt::LeftButton) {
    switch (mouseMode)
        {
        case MODE_ZOOM:
        setZoom(camera.zoom + 8 * dy);
        break;
        case MODE_PAN:
        setXTranslation(camera.xTran + 2 * -dx);
        setYTranslation(camera.yTran + 2 *  dy);
        break;
        case MODE_ROTATE:
        setZRotation(camera.zRot + 8 * -dy);
        break;
        }
    }
    lastPos = event->pos();
}

#ifdef __GNUC__
#  pragma GCC diagnostic pop
#endif

void
GLView2D::timerEvent (QTimerEvent *event)
{
    makeCurrent();

	// Window size.  Size may be zero if the window is not yet mapped.
	QSize s = size();

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


    _renderGL->getImage()->setPlane(getPlane(), getZ(), getC());
	_renderGL->getImage()->setMin(cmin);
	_renderGL->getImage()->setMax(cmax);

    GLWindow::timerEvent(event);

    renderLater();
}

