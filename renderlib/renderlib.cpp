#include "renderlib.h"

#include "glad/glad.h"
#include "HardwareWidget.h"
#include "Logging.h"
#include "ImageXyzcCuda.h"

#include <string>

//#include <QOpenGLWidget>
#include <QtGui/QOpenGLContext>
#include <QtGui/QOpenGLDebugLogger>
#include <QtGui/QWindow>
#include <QOffscreenSurface>

static bool renderLibInitialized = false;

static QOpenGLContext* dummyContext = nullptr;
static QOffscreenSurface* dummySurface = nullptr;

static QOpenGLDebugLogger* logger = nullptr;

std::map<std::string, std::shared_ptr<ImageCuda>> renderlib::sCudaImageCache;

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


	dummyContext = new QOpenGLContext();
	dummyContext->setFormat(format);    // ...and set the format on the context too
	dummyContext->create();
	LOG_INFO << "Created opengl context";

	dummySurface = new QOffscreenSurface();
	dummySurface->setFormat(dummyContext->format());
	dummySurface->create();
	LOG_INFO << "Created offscreen surface";
	bool ok = dummyContext->makeCurrent(dummySurface);
	if (!ok) {
		LOG_ERROR << "Failed to makeCurrent on offscreen surface";
	}
	else {
		LOG_INFO << "Made context current on offscreen surface";
	}


//	dummyWidget = new QOpenGLWidget();
//	dummyWidget->setMaximumSize(2, 2);
//	dummyWidget->show();
//	dummyWidget->hide();
//	dummyWidget->makeCurrent();

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

	// clean up the shared gpu cuda buffer cache
	for (auto i : sCudaImageCache) {
		i.second->deallocGpu();
	}
	sCudaImageCache.clear();

	delete dummySurface;
	dummySurface = nullptr;
	delete dummyContext;
	dummyContext = nullptr;
	delete logger;
	logger = nullptr;

	renderLibInitialized = false;
}

