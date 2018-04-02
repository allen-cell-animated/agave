#include "glad/glad.h"
#include "renderer.h"

#include "renderlib/CCamera.h"
#include "renderlib/FileReader.h"
#include "renderlib/HardwareWidget.h"
#include "renderlib/RenderGLCuda.h"
#include "renderlib/renderlib.h"
#include "renderlib/RenderSettings.h"
#include "renderlib/Logging.h"

#include "commandBuffer.h"
#include "command.h"

#include <QApplication>
#include <QElapsedTimer>
#include <QMessageBox>
#include <QOpenGLFramebufferObjectFormat>

Renderer::Renderer(QString id, QObject *parent) : QThread(parent),
	id(id), _streamMode(0), fbo(nullptr), _width(0), _height(0)
{
	this->totalQueueDuration = 0;

	LOG_DEBUG << "Renderer " << id.toStdString() << " -- Initializing rendering thread...";
	this->init();
	LOG_DEBUG << "Renderer " << id.toStdString() << " -- Done.";
}

Renderer::~Renderer()
{
	// delete all outstanding requests.
	qDeleteAll(this->requests);

	delete myVolumeData._renderSettings;
	delete myVolumeData._camera;
	delete myVolumeData._scene;
	delete myVolumeData._renderer;
}

void Renderer::myVolumeInit() {
	DeviceSelector d;

	myVolumeData._renderSettings = new RenderSettings();

	myVolumeData._camera = new CCamera();
	myVolumeData._camera->m_Film.m_ExposureIterations = 1;

	myVolumeData._scene = new Scene();

	myVolumeData._renderer = new RenderGLCuda(myVolumeData._renderSettings);
	myVolumeData._renderer->initialize(1024, 1024);
	myVolumeData._renderer->setScene(myVolumeData._scene);


}

void Renderer::init()
{
	//this->setFixedSize(1920, 1080);
	//QMessageBox::information(this, "Info:", "Application Directory: " + QApplication::applicationDirPath() + "\n" + "Working Directory: " + QDir::currentPath());

	QSurfaceFormat format;
	format.setSamples(16);    // Set the number of samples used for multisampling

	this->context = new QOpenGLContext();
	this->context->setFormat(format);    // ...and set the format on the context too
	this->context->create();

	this->surface = new QOffscreenSurface();
	this->surface->setFormat(this->context->format());
	this->surface->create();

	/*this->context->doneCurrent();
	this->context->moveToThread(this);*/
	this->context->makeCurrent(this->surface);

	int status = gladLoadGL();
	if (!status) {
		qDebug() << id << "COULD NOT LOAD GL ON THREAD";
	}

	///////////////////////////////////
	// INIT THE RENDER LIB
	///////////////////////////////////

	this->resizeGL(1024, 1024);


	int MaxSamples = 0;
	glGetIntegerv(GL_MAX_SAMPLES, &MaxSamples);
	qDebug() << id << "max samples" << MaxSamples;

	glEnable(GL_MULTISAMPLE);

	reset();

	this->context->doneCurrent();
	this->context->moveToThread(this);
}

void Renderer::run()
{
	this->context->makeCurrent(this->surface);

	// TODO: PUT THIS KIND OF INIT SOMEWHERE ELSE
	myVolumeInit();

	while (!QThread::currentThread()->isInterruptionRequested())
	{
		this->processRequest();

		QApplication::processEvents();
	}

	this->context->makeCurrent(this->surface);
	myVolumeData._renderer->cleanUpResources();
	shutDown();
}

void Renderer::addRequest(RenderRequest *request)
{
	this->requests << request;
	this->totalQueueDuration += request->getDuration();
}

bool Renderer::processRequest()
{
	if (this->requests.isEmpty())
	{
		return false;
	}

	//remove request from the queue
	RenderRequest *r = this->requests.takeFirst();
	this->totalQueueDuration -= r->getDuration();

	//process it
	QElapsedTimer timer;
	timer.start();

	std::vector<Command*> cmds = r->getParameters();
	if (cmds.size() > 0) {
		this->processCommandBuffer(cmds, r->getClient());
	}

	QImage img = this->render();

	r->setActualDuration(timer.nsecsElapsed());


	// in stream mode:
	// if queue is empty, then keep firing redraws back to client.
	// test about 100 frames as a convergence limit.
	if (_streamMode != 0 && this->requests.length() == 0 && myVolumeData._renderSettings->GetNoIterations() < 100) {
		// push another redraw request.
		std::vector<Command*> cmd;
		RequestRedrawCommandD data;
		cmd.push_back(new RequestRedrawCommand(data));
		this->addRequest(new RenderRequest(r->getClient(), cmd, false));
	}


	//inform the server that we are done with r
	emit requestProcessed(r, img);
	
	return true;
}

void Renderer::processCommandBuffer(std::vector<Command*>& cmds, QWebSocket* client)
{
	this->context->makeCurrent(this->surface);

	if (cmds.size() > 0) {
		ExecutionContext ec;
		ec._renderSettings = myVolumeData._renderSettings;
		ec._renderer = this;
		ec._appScene = myVolumeData._scene;
		ec._camera = myVolumeData._camera;
		ec._client = client;

		for (auto i = cmds.begin(); i != cmds.end(); ++i) {
			(*i)->execute(&ec);
		}

	}
}

QImage Renderer::render()
{
	this->context->makeCurrent(this->surface);

	glEnable(GL_TEXTURE_2D);

	// DRAW
	myVolumeData._camera->Update();
	myVolumeData._renderer->doRender(*(myVolumeData._camera));

	// COPY TO MY FBO
	this->fbo->bind();
	glViewport(0, 0, fbo->width(), fbo->height());
	myVolumeData._renderer->drawImage();
	this->fbo->release();

	QImage img = fbo->toImage();

	this->context->doneCurrent();

	return img;
}

void Renderer::resizeGL(int width, int height)
{
	if ((width == _width) && (height == _height)) {
		return;
	}

	this->context->makeCurrent(this->surface);

	// RESIZE THE RENDER INTERFACE
	if (myVolumeData._renderer) {
		myVolumeData._renderer->resize(width, height);
	}

	delete this->fbo;
	QOpenGLFramebufferObjectFormat fboFormat;
	fboFormat.setAttachment(QOpenGLFramebufferObject::CombinedDepthStencil);
	fboFormat.setMipmap(false);
	fboFormat.setSamples(0);
	fboFormat.setTextureTarget(GL_TEXTURE_2D);
	fboFormat.setInternalTextureFormat(GL_RGBA8);
	this->fbo = new QOpenGLFramebufferObject(width, height, fboFormat);

	glViewport(0, 0, width, height);

	_width = width;
	_height = height;
}

void Renderer::reset(int from)
{
	this->context->makeCurrent(this->surface);

	glClearColor(0.0, 0.0, 0.0, 1.0);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	//glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE);
	glEnable(GL_BLEND);
	glEnable(GL_LINE_SMOOTH);

	this->time.start();
	this->time = this->time.addMSecs(-from);
}

int Renderer::getTime()
{
	return this->time.elapsed();
}

void Renderer::shutDown()
{
	context->makeCurrent(surface);
	delete this->fbo;
	context->doneCurrent();
	delete context;

	// schedule this to be deleted only after we're done cleaning up
	surface->deleteLater();

	// Stop event processing, move the thread to GUI and make sure it is deleted.
	exit();
	moveToThread(QGuiApplication::instance()->thread());
}
