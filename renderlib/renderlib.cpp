#include "renderlib.h"

#include "glad/glad.h"
#include "HardwareWidget.h"
#include "Logging.h"

#include <string>

#include <QOpenGLWidget>
#include <QtGui/QOpenGLContext>
#include <QtGui/QOpenGLDebugLogger>
#include <QtGui/QWindow>

static bool renderLibInitialized = false;

static QOpenGLWidget* dummyWidget = nullptr;
static QOpenGLDebugLogger* logger = nullptr;

static const struct {
	int major = 3; 
	int minor = 3;
} AICS_GL_VERSION;

namespace {
	static void logMessage(QOpenGLDebugMessage message)
	{
		LOG_DEBUG << message.message().toStdString();
	}
}

int renderlib::initialize() {
	if (renderLibInitialized) {
		return 1;
	}
	renderLibInitialized = true;

	//boost::log::add_file_log("renderlib.log");
	LOG_INFO << "Renderlib startup";

	bool enableDebug = false;
	if (std::getenv("OME_QTWIDGETS_OPENGL_DEBUG"))
		enableDebug = true;

	QSurfaceFormat format;
	//format.setSamples(8);
	format.setDepthBufferSize(24);
	format.setStencilBufferSize(8);
	format.setVersion(AICS_GL_VERSION.major, AICS_GL_VERSION.minor);
	format.setProfile(QSurfaceFormat::CoreProfile);
	if (enableDebug) {
		format.setOption(QSurfaceFormat::DebugContext);
	}
	QSurfaceFormat::setDefaultFormat(format);

	dummyWidget = new QOpenGLWidget();
	dummyWidget->setMaximumSize(2, 2);
	dummyWidget->show();
	dummyWidget->hide();
	dummyWidget->makeCurrent();

	//glcontext = new QOpenGLContext();
	//bool valid = glcontext->create();
	//if (!valid) {
	//	LOG_ERROR << "Failed to create default OpenGL context";
	//}
	//bool ok = glcontext->makeCurrent(s);
	//	if (!ok) {
	//		LOG_ERROR << "Failed to makeCurrent default OpenGL context";
	//	}
	if (enableDebug)
	{
		logger = new QOpenGLDebugLogger();
		QObject::connect(logger, &QOpenGLDebugLogger::messageLogged,
			logMessage);
		if (logger->initialize())
		{
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

	DeviceSelector d;

	return status;
}

void renderlib::cleanup() {
	if (!renderLibInitialized) {
		return;
	}
	LOG_INFO << "Renderlib shutdown";

	delete dummyWidget;
	dummyWidget = nullptr;
	delete logger;
	logger = nullptr;

	renderLibInitialized = false;
}

