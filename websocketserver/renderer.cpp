#include "glad/glad.h"
#include "renderer.h"

#include "renderlib/FileReader.h"
#include "renderlib/HardwareWidget.h"
#include "renderlib/RenderGLCuda.h"
#include "renderlib/renderlib.h"
#include "renderlib/Scene.h"

#include "commandBuffer.h"
#include "command.h"

#include <QApplication>
#include <QElapsedTimer>
#include <QMessageBox>
#include <QOpenGLFramebufferObjectFormat>

Renderer::Renderer(QString id, QObject *parent) : QThread(parent),
id(id), _streamMode(0)
{
	this->totalQueueDuration = 0;

	qDebug() << id << "Initializing rendering thread...";
	this->init();
	qDebug() << id << "Done.";
}

Renderer::~Renderer()
{
}

void Renderer::myVolumeInit() {
	DeviceSelector d;

	FileReader fileReader;
	std::string file("C:\\Users\\danielt.ALLENINST\\Downloads\\AICS-12_269_4.ome.tif");
	std::shared_ptr<ImageXYZC> image = fileReader.loadOMETiff_4D(file);
	myVolumeData._image = image;
	myVolumeData._scene = new CScene();
	myVolumeData._scene->m_Camera.m_Film.m_ExposureIterations = 1;
	myVolumeData._scene->m_DiffuseColor[0] = 1.0;
	myVolumeData._scene->m_DiffuseColor[1] = 1.0;
	myVolumeData._scene->m_DiffuseColor[2] = 1.0;
	myVolumeData._scene->m_DiffuseColor[3] = 1.0;

	myVolumeData._renderer = new RenderGLCuda(image, myVolumeData._scene);
	myVolumeData._renderer->initialize(1024, 1024);
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

	QOpenGLFramebufferObjectFormat fboFormat;
	fboFormat.setAttachment(QOpenGLFramebufferObject::CombinedDepthStencil);
	fboFormat.setMipmap(false);
	fboFormat.setSamples(16);
	fboFormat.setTextureTarget(GL_TEXTURE_2D);
	fboFormat.setInternalTextureFormat(GL_RGBA32F_ARB);
	this->fbo = new QOpenGLFramebufferObject(512, 512, fboFormat);

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

	while (1)
	{
		this->processRequest();

		if (_streamMode) {

		}
		QApplication::processEvents();
	}
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
		this->processCommandBuffer(cmds);
	}

	QImage img = this->render();

	r->setActualDuration(timer.nsecsElapsed());

	//inform the server
	emit requestProcessed(r, img);

	return true;
}

void Renderer::processCommandBuffer(std::vector<Command*>& cmds)
{
	this->context->makeCurrent(this->surface);

	if (cmds.size() > 0) {
		ExecutionContext ec;
		ec._scene = myVolumeData._scene;
		ec._renderer = this;

		for (auto i = cmds.begin(); i != cmds.end(); ++i) {
			(*i)->execute(&ec);
		}

	}
}

QImage Renderer::render()
{
	this->context->makeCurrent(this->surface);

	glEnable(GL_TEXTURE_2D);

	// TODO these should be commands, and not part of "render()"
#if 0
	if ((p.mseDx != 0) || (p.mseDy != 0)) {
		myVolumeData._scene->m_Camera.Orbit(-0.6f * (float)(p.mseDy), -(float)(p.mseDx));
		myVolumeData._scene->SetNoIterations(0);
	}
	if (p.channelvalues[0].toInt() != myVolumeData._scene->_channel) {
		myVolumeData._scene->_channel = p.channelvalues[0].toInt();
		myVolumeData._scene->SetNoIterations(0);
	}
#endif

	// DRAW THE THINGS INTO THEIR OWN FBOs
	myVolumeData._renderer->doRender();
	//		foreach(SceneDescription scene, this->scenes)
	//		{
	//			this->renderScene(scene.name);
	//		}

	// BIND THE RENDER TARGET FOR THE FINAL IMAGE
	// {
	//		this->marion->fbo("_")->bind();
	//		glClearColor(1.0, 1.0, 1.0, 1.0);
	//		glClear(GL_COLOR_BUFFER_BIT);
	// COMPOSITE THE SCENE'S FBO TO THE FINAL IMAGE FBO
	//		foreach(SceneDescription scene, this->scenes)
	//		{
	//			this->displayScene(scene.name);
	//		}
	// UNBIND SO WE CAN READ THE TARGET
	//		this->marion->fbo("_")->release();
	// }
	//qDebug() << "gu" << this->marion->fbo("_")->toImage().width() << this->marion->fbo("_")->toImage().height();
	//QOpenGLFramebufferObject::blitFramebuffer(fbo, QRect(0, 0, 512, 512), this->marion->fbo("_"), QRect(0, 0, 512, 512), GL_COLOR_BUFFER_BIT, GL_LINEAR);

	// DRAW QUAD TO FBO (COPY RENDERED FBO TO PRIMARY FBO)
	// try glBlitFramebuffer() instead?
	this->fbo->bind();
	glViewport(0, 0, fbo->width(), fbo->height());

	myVolumeData._renderer->drawImage();

	glEnable(GL_TEXTURE_2D);
	this->fbo->release();


	QImage img = fbo->toImage();

	this->context->doneCurrent();

	return img;
}

void Renderer::renderScene(QString scene)
{
}

void Renderer::displayScene(QString scene)
{
}

void Renderer::resizeGL(int width, int height)
{
	this->context->makeCurrent(this->surface);

	// RESIZE THE RENDER INTERFACE
	if (myVolumeData._renderer) {
		myVolumeData._renderer->resize(width, height);
	}

	int w, h;
	w = width;
	h = (int)((GLfloat)width * 0.5625);

	glViewport(0, 0, width, height);

	//~ projection matrix setup, all the view plugins should use this one for their transformations
	projection.setToIdentity();
	int internalWidth=w, internalHeight=h;
	projection.perspective(20.0, (qreal) internalWidth / (qreal) internalHeight, 0.1f, 16.0f);
}

void Renderer::reset(int from)
{
	this->context->makeCurrent(this->surface);

	glClearColor(0.0, 0.0, 0.0, 1.0);
	glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE);
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
