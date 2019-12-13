#include "renderer.h"

#include "renderlib/CCamera.h"
#include "renderlib/FileReader.h"
#include "renderlib/Logging.h"
#include "renderlib/RenderGLPT.h"
#include "renderlib/RenderSettings.h"
#include "renderlib/renderlib.h"

#include "command.h"
#include "commandBuffer.h"

#include <QApplication>
#include <QElapsedTimer>
#include <QMessageBox>
#include <QOpenGLFramebufferObjectFormat>

Renderer::Renderer(QString id, QObject* parent, QMutex& mutex)
  : QThread(parent)
  , id(id)
  , _streamMode(0)
  , fbo(nullptr)
  , _width(0)
  , _height(0)
  , _openGLMutex(&mutex)
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
}

void
Renderer::myVolumeInit()
{
  static const int initWidth = 1024, initHeight = 1024;

  myVolumeData._renderSettings = new RenderSettings();

  myVolumeData._camera = new CCamera();
  myVolumeData._camera->m_Film.m_ExposureIterations = 1;
  myVolumeData._camera->m_Film.m_Resolution.SetResX(initWidth);
  myVolumeData._camera->m_Film.m_Resolution.SetResY(initHeight);

  myVolumeData._scene = new Scene();
  myVolumeData._scene->initLights();

  myVolumeData._renderer = new RenderGLPT(myVolumeData._renderSettings);
  myVolumeData._renderer->initialize(initWidth, initHeight);
  myVolumeData._renderer->setScene(myVolumeData._scene);
}

void
Renderer::init()
{
  // this->setFixedSize(1920, 1080);
  // QMessageBox::information(this, "Info:", "Application Directory: " + QApplication::applicationDirPath() + "\n" +
  // "Working Directory: " + QDir::currentPath());

  QSurfaceFormat format;
  format.setSamples(16); // Set the number of samples used for multisampling

  this->context = new QOpenGLContext();
  this->context->setFormat(format); // ...and set the format on the context too
  this->context->create();

  this->surface = new QOffscreenSurface();
  this->surface->setFormat(this->context->format());
  this->surface->create();

  /*this->context->doneCurrent();
  this->context->moveToThread(this);*/
  this->context->makeCurrent(this->surface);

  int status = gladLoadGL();
  if (!status) {
    LOG_INFO << id.toStdString() << "COULD NOT LOAD GL ON THREAD";
  }

  ///////////////////////////////////
  // INIT THE RENDER LIB
  ///////////////////////////////////

  this->resizeGL(1024, 1024);

  int MaxSamples = 0;
  glGetIntegerv(GL_MAX_SAMPLES, &MaxSamples);
  LOG_INFO << id.toStdString() << "max samples" << MaxSamples;

  glEnable(GL_MULTISAMPLE);

  reset();

  this->context->doneCurrent();
  this->context->moveToThread(this);
}

void
Renderer::run()
{
  this->context->makeCurrent(this->surface);

  // TODO: PUT THIS KIND OF INIT SOMEWHERE ELSE
  myVolumeInit();

  while (!QThread::currentThread()->isInterruptionRequested()) {
    this->processRequest();

    QApplication::processEvents();
  }

  this->context->makeCurrent(this->surface);
  myVolumeData._renderer->cleanUpResources();
  shutDown();
}

void
Renderer::addRequest(RenderRequest* request)
{
  this->requests << request;
  this->totalQueueDuration += request->getDuration();
}

bool
Renderer::processRequest()
{
  if (this->requests.isEmpty()) {
    return false;
  }

  if (_streamMode != 0) {
    QElapsedTimer timer;
    timer.start();

    RenderRequest* lastReq = nullptr;

    // eat requests until done, and then render
    // note that any one request could change the streaming mode.
    while (!this->requests.isEmpty() && _streamMode != 0) {

      RenderRequest* r = this->requests.takeFirst();
      this->totalQueueDuration -= r->getDuration();

      std::vector<Command*> cmds = r->getParameters();
      if (cmds.size() > 0) {
        this->processCommandBuffer(r);
      }

      // the true last request will be passed to "emit" and deleted later
      if (!this->requests.isEmpty() && _streamMode != 0) {
        delete r;
      } else {
        lastReq = r;
      }
    }

    QWebSocket* ws = lastReq->getClient();
    LOG_DEBUG << "RENDER for " << ws->peerName().toStdString() << "(" << ws->peerAddress().toString().toStdString()
              << ":" << QString::number(ws->peerPort()).toStdString() << ")";

    QImage img = this->render();

    lastReq->setActualDuration(timer.nsecsElapsed());

    // in stream mode:
    // if queue is empty, then keep firing redraws back to client.
    // test about 100 frames as a convergence limit.
    if (_streamMode != 0 && myVolumeData._renderSettings->GetNoIterations() < 500) {
      // push another redraw request.
      std::vector<Command*> cmd;
      RequestRedrawCommandD data;
      cmd.push_back(new RequestRedrawCommand(data));
      this->addRequest(new RenderRequest(lastReq->getClient(), cmd, false));
    }

    // inform the server that we are done with r
    emit requestProcessed(lastReq, img);

  } else {
    // if not in stream mode, then process one request, then re-render.

    // remove request from the queue
    RenderRequest* r = this->requests.takeFirst();
    this->totalQueueDuration -= r->getDuration();

    // process it
    QElapsedTimer timer;
    timer.start();

    std::vector<Command*> cmds = r->getParameters();
    if (cmds.size() > 0) {
      this->processCommandBuffer(r);
    }

    QImage img = this->render();

    r->setActualDuration(timer.nsecsElapsed());

    // inform the server that we are done with r
    emit requestProcessed(r, img);
  }

  return true;
}

void
Renderer::processCommandBuffer(RenderRequest* rr)
{
  this->context->makeCurrent(this->surface);

  std::vector<Command*> cmds = rr->getParameters();
  if (cmds.size() > 0) {
    ExecutionContext ec;
    ec.m_renderSettings = myVolumeData._renderSettings;
    ec.m_renderer = this;
    ec.m_appScene = myVolumeData._scene;
    ec.m_camera = myVolumeData._camera;
    ec.m_message = "";

    for (auto i = cmds.begin(); i != cmds.end(); ++i) {
      (*i)->execute(&ec);
      if (!ec.m_message.isEmpty()) {
        emit sendString(rr, ec.m_message);
        ec.m_message = "";
      }
    }
  }
}

QImage
Renderer::render()
{
  _openGLMutex->lock();
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
  _openGLMutex->unlock();

  return img;
}

void
Renderer::resizeGL(int width, int height)
{
  if ((width == _width) && (height == _height)) {
    return;
  }
  _openGLMutex->lock();

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

  _openGLMutex->unlock();
}

void
Renderer::reset(int from)
{
  _openGLMutex->lock();

  this->context->makeCurrent(this->surface);

  glClearColor(0.0, 0.0, 0.0, 1.0);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  // glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE);
  glEnable(GL_BLEND);
  glEnable(GL_LINE_SMOOTH);

  this->time.start();
  this->time = this->time.addMSecs(-from);

  _openGLMutex->unlock();
}

int
Renderer::getTime()
{
  return this->time.elapsed();
}

void
Renderer::shutDown()
{
  context->makeCurrent(surface);
  delete this->fbo;

  delete myVolumeData._renderSettings;
  delete myVolumeData._camera;
  delete myVolumeData._scene;
  delete myVolumeData._renderer;
  myVolumeData._camera = nullptr;
  myVolumeData._scene = nullptr;
  myVolumeData._renderSettings = nullptr;
  myVolumeData._renderer = nullptr;

  context->doneCurrent();
  delete context;

  // schedule this to be deleted only after we're done cleaning up
  surface->deleteLater();

  // Stop event processing, move the thread to GUI and make sure it is deleted.
  exit();
  moveToThread(QGuiApplication::instance()->thread());
}
