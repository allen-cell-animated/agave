#include "glad/glad.h"
#include "renderer.h"
#include <QApplication>
#include <QElapsedTimer>
#include <QMessageBox>
#include <QOpenGLFramebufferObject>

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

	QOpenGLFramebufferObjectFormat fboFormat;
	fboFormat.setAttachment(QOpenGLFramebufferObject::CombinedDepthStencil);
	fboFormat.setMipmap(false);
	fboFormat.setSamples(16);
	fboFormat.setTextureTarget(GL_TEXTURE_2D);
	fboFormat.setInternalTextureFormat(GL_RGBA32F_ARB);
	this->fbo = new QOpenGLFramebufferObject(512, 512, fboFormat);


	//this->fbo = new QOpenGLFramebufferObject(512, 512, fboFormat);

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


	// qDebug() << id << "Extensions===========================";
	// foreach (QByteArray extension, this->marion->getContext()->extensions())
	// {
	// 	qDebug() << QString(extension);
	// }
	// qDebug() << id << "/Extensions===========================";

	//cell setup
	//connect(this->marion->getObject(), SIGNAL(shaderRecompiled()), this, SLOT(update()));

	//glEnable(GL_MULTISAMPLE);

	reset();

	this->context->doneCurrent();
	this->context->moveToThread(this);
}

void Renderer::run()
{
	this->context->makeCurrent(this->surface);

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

	QImage img = this->render(r->getParameters());

	r->setActualDuration(timer.nsecsElapsed());

	//inform the server
	emit requestProcessed(r, img);

	return true;
}

QImage Renderer::render(RenderParameters p)
{	
	this->context->makeCurrent(this->surface);

	glEnable(GL_TEXTURE_2D);

	QList<QVariant> cellParams;
	cellParams << QVariant(projection)
			   << QVariant(p.modelview)
			   << QVariant(p.visibility)
			   << QVariant(p.mitoFuzziness);

	//this->marion->library("cell")->getInterface()->setParameters(cellParams);

	//this->marion->fbo("_")->bind();
	glClearColor(1.0, 1.0, 1.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT);
	//glColor4f(1.0, 1.0, 1.0, 1.0);

	// foreach (SceneDescription scene, this->scenes)
	// {
	// 	glColor4f(1.0, 1.0, 1.0, 1.0);
	// 	this->displayScene(scene.name);
	// }
	// this->marion->fbo("_")->release();

	//qDebug() << "gu" << this->marion->fbo("_")->toImage().width() << this->marion->fbo("_")->toImage().height();

	//QOpenGLFramebufferObject::blitFramebuffer(fbo, QRect(0, 0, 512, 512), this->marion->fbo("_"), QRect(0, 0, 512, 512), GL_COLOR_BUFFER_BIT, GL_LINEAR);


	this->fbo->bind();
		glClearColor(0.0, 0.0, 0.0, 1.0);
		glClear(GL_COLOR_BUFFER_BIT);

		glEnable(GL_TEXTURE_2D);
//		glBindTexture(GL_TEXTURE_2D, this->marion->fbo("_")->texture());
		//glBegin(GL_QUADS);
		//	glColor4f(1.0, 1.0, 1.0, 1.0);
		//	glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
		//	glTexCoord2f(1.0, 0.0); glVertex2f(+0.0, -1.0);
		//	glTexCoord2f(1.0, 1.0); glVertex2f(+0.0, +0.0);
		//	glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, +0.0);
		//glEnd();
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
	// DynamicLibrary *lib = this->marion->library(scene);

	// if (lib == 0)
	// {
	// 	return;
	// }
	// else
	// {
	// 	int t = getTime();
	// 	if (lib->getInterface()->isPlaying(t))
	// 	{
	// 		lib->getInterface()->render(t);
	// 	}
	// }
}

void Renderer::displayScene(QString scene)
{
	// DynamicLibrary *lib = this->marion->library(scene);

	// if (lib == 0)
	// {
	// 	return;
	// }
	// else if (lib->getInterface()->isPlaying(getTime()))
	// {
	// 	glColor4f(1.0, 1.0, 1.0, 1.0);
	// 	glBindTexture(GL_TEXTURE_2D, this->marion->fbo(scene)->texture());
	// 	this->marion->drawFullscreenRect();
	// }
}

void Renderer::resizeGL(int width, int height)
{
	this->context->makeCurrent(this->surface);

//	this->marion->resizeGL(width, height);

	int w, h;
	w = width;
	h = (int) ((GLfloat) width * 0.5625);

	//glViewport(0, (height - h) / 2, w, h);
	glViewport(0, 0, width, height);

	//~ projection matrix setup, all the view plugins should use this one for their transformations
	projection.setToIdentity();
//	projection.perspective(20.0, (qreal) this->marion->internalWidth() / (qreal) this->marion->internalHeight(), 0.1, 16.0);


	//glMatrixMode(GL_PROJECTION);
	//glLoadIdentity();
	//glOrtho(-1.0, 1.0, -1.0, 1.0, 0.0, 1.0);
	//glClear(GL_ACCUM_BUFFER_BIT);
	//glMatrixMode(GL_MODELVIEW);

	//this->update();
}

void Renderer::reset(int from)
{
	this->context->makeCurrent(this->surface);

	glClearColor(0.0,0.0,0.0,1.0);
//	this->marion->gl()->glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE);
	glEnable(GL_BLEND);
	glEnable(GL_LINE_SMOOTH);

	//glClear(GL_ACCUM_BUFFER_BIT);

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
