#include <cstdlib>
#include <iostream>

#include "GLWindow.h"

#include "renderlib/renderlib.h"

#include <QtCore/QCoreApplication>

#include <QtGui/QOpenGLContext>
#include <QtGui/QOpenGLDebugLogger>
#include <QtGui/QOpenGLPaintDevice>
#include <QtGui/QPainter>

GLWindow::GLWindow(QWindow *parent):
    QWindow(parent),
    m_update_pending(false),
    m_animating(false),
    m_glcontext(0),
    m_device(0),
    m_logger(0)
{
    setSurfaceType(QWindow::OpenGLSurface);
}

GLWindow::~GLWindow()
{
    if (m_logger)
		m_logger->stopLogging();
    delete m_device;
}

void
GLWindow::render(QPainter *painter)
{
    Q_UNUSED(painter);
}

void
GLWindow::initialize()
{
}

void
GLWindow::resize()
{
}

void
GLWindow::render()
{
    if (!m_device)
	    m_device = new QOpenGLPaintDevice;

//      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

    m_device->setSize(size());

    QPainter painter(m_device);
    render(&painter);
}

void
GLWindow::renderLater()
{
    if (!m_update_pending) {
		m_update_pending = true;
		QCoreApplication::postEvent(this, new QEvent(QEvent::UpdateRequest));
    }
}

bool
GLWindow::event(QEvent *event)
{
    switch (event->type()) {
		case QEvent::UpdateRequest:
			m_update_pending = false;
			renderNow();
			return true;
		default:
			return QWindow::event(event);
    }
}

void
GLWindow::exposeEvent(QExposeEvent *event)
{
    Q_UNUSED(event);

    if (isExposed())
		renderNow();
}

void GLWindow::resizeEvent(QResizeEvent * /* event */)
{
    if (m_glcontext)
		resize();
}

QOpenGLContext *
GLWindow::context() const
{
    return m_glcontext;
}

void
GLWindow::makeCurrent()
{
    if (m_glcontext)
		m_glcontext->makeCurrent(this);
}

void GLWindow::renderNow()
{
    if (!isExposed())
		return;

    bool needsInitialize = false;
    bool enableDebug = false;

    if (std::getenv("OME_QTWIDGETS_OPENGL_DEBUG"))
		enableDebug = true;

    if (!m_glcontext) {
		m_glcontext = new QOpenGLContext(this);
		bool valid = m_glcontext->create();
		std::cerr << "Valid OpenGL context: " << valid << std::endl;
		makeCurrent();

		if (enableDebug)
        {
			m_logger = new QOpenGLDebugLogger(this);
			connect(m_logger, SIGNAL(messageLogged(QOpenGLDebugMessage)),
                this, SLOT(logMessage(QOpenGLDebugMessage)),
                Qt::DirectConnection);
			if (m_logger->initialize())
            {
				m_logger->startLogging(QOpenGLDebugLogger::SynchronousLogging);
				m_logger->enableMessages();
            }
        }

		needsInitialize = true;
    }

    makeCurrent();

    if (needsInitialize)
    {
        initialize();
    }

    render();

    m_glcontext->swapBuffers(this);

    if (m_animating)
		renderLater();
}

void
GLWindow::setAnimating(bool animating)
{
    this->m_animating = animating;

    if (this->m_animating)
		renderLater();
}

void
GLWindow::logMessage(QOpenGLDebugMessage message)
{
    std::cerr << message.message().toStdString();
}
