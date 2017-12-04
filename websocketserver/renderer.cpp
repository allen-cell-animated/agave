#include "glad/glad.h"
#include "renderer.h"

#include "renderlib/FileReader.h"
#include "renderlib/HardwareWidget.h"
#include "renderlib/RenderGLCuda.h"
#include "renderlib/renderlib.h"
#include "renderlib/Scene.h"
#include "renderlib/command.h"

#include "commandBuffer.h"

#include <QApplication>
#include <QElapsedTimer>
#include <QMessageBox>
#include <QOpenGLFramebufferObjectFormat>

Renderer::Renderer(QString id, QObject *parent) : QThread(parent),
id(id)
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


	//this->fbo = new QOpenGLFramebufferObject(512, 512, fboFormat);

	///////////////////////////////////
	// INIT THE RENDER LIB
	///////////////////////////////////
//	this->marion = new Marion(this->context);
//	this->marion->resizeGL(512, 512);
//	this->marion->initializeGL();

	this->resizeGL(1024, 1024);


	int MaxSamples = 0;
	glGetIntegerv(GL_MAX_SAMPLES, &MaxSamples);
	qDebug() << id << "max samples" << MaxSamples;

	/*this->fbo->bind();

	glClearColor(1.0, 0.7, 1.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);

	this->fbo->release();

	this->fbo->toImage().save("offscreen.png");*/


	//this->context->doneCurrent();

#if 0
	qDebug() << id << "Extensions===========================";
	foreach(QByteArray extension, this->marion->getContext()->extensions())
	{
		qDebug() << QString(extension);
	}
	qDebug() << id << "/Extensions===========================";

	//cell setup
	connect(this->marion->getObject(), SIGNAL(shaderRecompiled()), this, SLOT(update()));


	qDebug() << id << "Loading Scenes...";
	//scenes

	this->scenes << SceneDescription("cellLoader", 0, 0)
		<< SceneDescription("cell", 0, 0);

	DynamicLibrary::setPath("../scenes/");
	foreach(SceneDescription scene, this->scenes)
	{
		//this->marion->addLibrary(scene.name, new DynamicLibrary(scene.name + "/build/" + scene.name + ".dll", scene.start, scene.end, this->marion));

#ifdef __linux__
		this->marion->addLibrary(scene.name, new DynamicLibrary(scene.name + "/build/lib" + scene.name.toLower() + ".so", scene.start, scene.end, this->marion));
#elif _WIN32
#ifdef _DEBUG
		//this->marion->addLibrary(scene.name, new DynamicLibrary(scene.name + "/build/debug/" + scene.name + ".dll", scene.start, scene.end, this->marion));
		this->marion->addLibrary(scene.name, new DynamicLibrary(scene.name + "/build/" + scene.name + ".dll", scene.start, scene.end, this->marion));
#else
		//this->marion->addLibrary(scene.name, new DynamicLibrary(scene.name + "/build/release/" + scene.name + ".dll", scene.start, scene.end, this->marion));
		bool deploy = true;
		this->marion->addLibrary(scene.name, new DynamicLibrary((!deploy ? (scene.name + "/build/") : ("")) + scene.name + ".dll", scene.start, scene.end, this->marion));
#endif

#endif

		this->marion->addFbo(scene.name);
	}
#endif 

	glEnable(GL_MULTISAMPLE);

	reset();

	this->context->doneCurrent();
	this->context->moveToThread(this);
}

void Renderer::run()
{
	this->context->makeCurrent(this->surface);
	myVolumeInit();

//	int status = gladLoadGL();
//	if (!status) {
//		qDebug() << id << "COULD NOT LOAD GL ON THREAD";
//	}

	while (1)
	{
		this->processRequest();

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

	if (r->getParameters()._cmds.size() > 0) {
		this->processCommandBuffer(r->getParameters()._cmds);
	}
	else {
		QImage img = this->render(r->getParameters());

		r->setActualDuration(timer.nsecsElapsed());

		//inform the server
		emit requestProcessed(r, img);
	}

	return true;
}

void Renderer::processCommandBuffer(std::vector<Command*>& cmds)
{
	this->context->makeCurrent(this->surface);

	if (cmds.size() > 0) {
		ExecutionContext ec;
		ec._scene = myVolumeData._scene;
		for (auto i = cmds.begin(); i != cmds.end(); ++i) {
			(*i)->execute(&ec);
		}

	}
}

QImage Renderer::render(RenderParameters p)
{
	this->context->makeCurrent(this->surface);

	glEnable(GL_TEXTURE_2D);
#if 0
	//todo client: update render params
	QList<QVariant> cellParams;
	cellParams << QVariant(projection)
		<< QVariant(p.modelview)
		<< QVariant(p.visibility)
		<< QVariant(p.mitoFuzziness)
		<< QVariant(p.type1)
		<< QVariant(p.type2)
		<< QVariant(p.cell1)
		<< QVariant(p.cell2)
		<< QVariant(p.mode)
		<< QVariant(p.crossFade)
		<< QVariant(p.channelvalues)
		<< QVariant(p.usingCellServer);

	this->marion->library("cell")->getInterface()->setParameters(cellParams);
#endif
	if ((p.mseDx != 0) || (p.mseDy != 0)) {
		myVolumeData._scene->m_Camera.Orbit(-0.6f * (float)(p.mseDy), -(float)(p.mseDx));
		myVolumeData._scene->SetNoIterations(0);
	}
	if (p.channelvalues[0].toInt() != myVolumeData._scene->_channel) {
		myVolumeData._scene->_channel = p.channelvalues[0].toInt();
		myVolumeData._scene->SetNoIterations(0);
	}


	// DRAW THE THINGS INTO THEIR OWN FBOs
	myVolumeData._renderer->doRender();

	// BIND THE RENDER TARGET FOR THE FINAL IMAGE
	//this->marion->fbo("_")->bind();
//	glClearColor(1.0, 1.0, 1.0, 1.0);
//	glClear(GL_COLOR_BUFFER_BIT);

	// COMPOSITE THE SCENE'S FBO TO THE FINAL IMAGE FBO
	foreach(SceneDescription scene, this->scenes)
	{
		this->displayScene(scene.name);
	}
	// UNBIND SO WE CAN READ THE TARGET
	//this->marion->fbo("_")->release();

	//qDebug() << "gu" << this->marion->fbo("_")->toImage().width() << this->marion->fbo("_")->toImage().height();

	//QOpenGLFramebufferObject::blitFramebuffer(fbo, QRect(0, 0, 512, 512), this->marion->fbo("_"), QRect(0, 0, 512, 512), GL_COLOR_BUFFER_BIT, GL_LINEAR);

	// DRAW QUAD TO FBO (COPY RENDERED FBO TO PRIMARY FBO)
	// try glBlitFramebuffer() instead?
	this->fbo->bind();
	glViewport(0, 0, fbo->width(), fbo->height());

	myVolumeData._renderer->drawImage();
//	glClearColor(0.0, 0.0, 0.0, 1.0);
//	glClear(GL_COLOR_BUFFER_BIT);

//	glEnable(GL_TEXTURE_2D);
	// BIND THE FRAME RESULTS AS A TEXTURE
//	glBindTexture(GL_TEXTURE_2D, myVolumeData._renderer->getFboTexture());
//GL	glBegin(GL_QUADS);
//GL	glColor4f(1.0, 1.0, 1.0, 1.0);
//GL	glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
//GL	glTexCoord2f(1.0, 0.0); glVertex2f(+0.0, -1.0);
//GL	glTexCoord2f(1.0, 1.0); glVertex2f(+0.0, +0.0);
//GL	glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, +0.0);
//GL	glEnd();
	glEnable(GL_TEXTURE_2D);
	this->fbo->release();


	//QImage img = this->marion->fbo("_")->toImage();
	QImage img = fbo->toImage();
	//QImage img = fbo->toImage().scaled(512, 256, Qt::IgnoreAspectRatio, Qt::SmoothTransformation);

	this->context->doneCurrent();

	return img;


	/*glEnable(GL_BLEND);
	this->marion->gl()->glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE);
	glEnable(GL_TEXTURE_2D);

	glClearColor(1,1,1,1);
	glClear(GL_COLOR_BUFFER_BIT);
	this->marion->bindTexture(GL_TEXTURE_2D, this->marion->fbo("_")->texture(), GL_TEXTURE0);
	this->marion->drawFullscreenRect();*/




	//this->fbo("___")->toImage().save("video/"+QString::number(frameNumber)+".png");
	//this->grabFramebuffer().save("video/"+QString::number(11000 + frameNumber)+".png");
	//frameNumber++;
	//fakeTime+=20;

	//update();
}

void Renderer::renderScene(QString scene)
{
#if 0
	DynamicLibrary *lib = this->marion->library(scene);

	if (lib == 0)
	{
		return;
	}
	else
	{
		int t = getTime();
		if (lib->getInterface()->isPlaying(t))
		{
			//qDebug() << scene;
			//this->fbo(scene)->bind();
			lib->getInterface()->render(t);
			//this->fbo(scene)->release();
		}
	}
#endif
}

void Renderer::displayScene(QString scene)
{
#if 0
	DynamicLibrary *lib = this->marion->library(scene);

	if (lib == 0)
	{
		return;
	}
	else if (lib->getInterface()->isPlaying(getTime()))
	{
		glColor4f(1.0, 1.0, 1.0, 1.0);
		glBindTexture(GL_TEXTURE_2D, this->marion->fbo(scene)->texture());
		this->marion->drawFullscreenRect();
	}
#endif
}

void Renderer::resizeGL(int width, int height)
{
	this->context->makeCurrent(this->surface);

	// RESIZE THE RENDER INTERFACE
	//this->marion->resizeGL(width, height);
	if (myVolumeData._renderer) {
		myVolumeData._renderer->resize(width, height);
	}

	int w, h;
	w = width;
	h = (int)((GLfloat)width * 0.5625);

	//glViewport(0, (height - h) / 2, w, h);
	glViewport(0, 0, width, height);

	//~ projection matrix setup, all the view plugins should use this one for their transformations
	projection.setToIdentity();
	int internalWidth=w, internalHeight=h;
	projection.perspective(20.0, (qreal) internalWidth / (qreal) internalHeight, 0.1f, 16.0f);
	//projection.perspective(20.0, (qreal) this->marion->internalWidth() / (qreal) this->marion->internalHeight(), 0.1, 16.0);


//GL	glMatrixMode(GL_PROJECTION);
//GL	glLoadIdentity();
//GL	glOrtho(-1.0, 1.0, -1.0, 1.0, 0.0, 1.0);
//GL	glClear(GL_ACCUM_BUFFER_BIT);
//GL	glMatrixMode(GL_MODELVIEW);

	//this->update();
}

void Renderer::reset(int from)
{
	this->context->makeCurrent(this->surface);

	glClearColor(0.0, 0.0, 0.0, 1.0);
	glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE);
	glEnable(GL_BLEND);
	glEnable(GL_LINE_SMOOTH);

//	glClear(GL_ACCUM_BUFFER_BIT);

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
