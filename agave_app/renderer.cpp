#include "renderer.h"

#include "renderlib/AppScene.h"
#include "renderlib/CCamera.h"
#include "renderlib/Logging.h"
#include "renderlib/RenderSettings.h"
#include "renderlib/ScaleBarTool.h"
#include "renderlib/SceneView.h"
#include "renderlib/graphics/RenderGL.h"
#include "renderlib/graphics/RenderGLPT.h"
#include "renderlib/io/FileReader.h"

#include "command.h"
#include "commandBuffer.h"

#include <QApplication>
#include <QElapsedTimer>
#include <QMessageBox>
#include <QMutexLocker>
#include <QOpenGLFramebufferObjectFormat>

Renderer::Renderer(QString id, QObject* parent, QMutex& mutex)
  : QThread(parent)
  , m_id(id)
  , m_streamMode(false)
  , m_fbo(nullptr)
  , m_width(0)
  , m_height(0)
  , m_openGLMutex(&mutex)
  , m_wait()
  , m_rglContext()
{
  this->m_totalQueueDuration = 0;

  LOG_DEBUG << "Renderer " << id.toStdString() << " -- Initializing rendering thread...";
  LOG_DEBUG << "Renderer " << id.toStdString() << " -- Done.";
}

Renderer::~Renderer()
{
  // delete all outstanding requests.
  qDeleteAll(this->m_requests);
}

void
Renderer::configure(IRenderWindow* renderer,
                    const RenderSettings& renderSettings,
                    const Scene& scene,
                    const CCamera& camera,
                    const LoadSpec& loadSpec,
                    // rendererMode ignored if renderer is non-null
                    renderlib::RendererType rendererMode,
                    QOpenGLContext* glContext)
{
  // assumes scene is already set in renderer and everything is initialized
  m_myVolumeData.m_renderSettings = new RenderSettings(renderSettings);
  m_myVolumeData.m_camera = new CCamera(camera);
  m_myVolumeData.m_scene = new Scene(scene);
  m_myVolumeData.m_loadSpec = loadSpec;
  m_ec.m_loadSpec = loadSpec;
  if (!renderer) {
    m_myVolumeData.m_camera->m_Film.m_Resolution.SetResX(1024);
    m_myVolumeData.m_camera->m_Film.m_Resolution.SetResY(1024);

    m_myVolumeData.ownRenderer = true;
    m_myVolumeData.m_renderer = renderlib::createRenderer(rendererMode, m_myVolumeData.m_renderSettings);
    m_myVolumeData.m_renderer->setScene(m_myVolumeData.m_scene);
  } else {
    m_myVolumeData.ownRenderer = false;
    m_myVolumeData.m_renderer = renderer;
  }

  m_rglContext.configure(glContext);
}

void
Renderer::init()
{
  m_rglContext.init();

  int status = gladLoadGL();
  if (!status) {
    LOG_INFO << m_id.toStdString() << " COULD NOT LOAD GL ON THREAD";
  }

  ///////////////////////////////////
  // INIT THE RENDER LIB
  ///////////////////////////////////
  if (m_myVolumeData.ownRenderer) {
    m_myVolumeData.m_renderer->initialize(m_myVolumeData.m_camera->m_Film.m_Resolution.GetResX(),
                                          m_myVolumeData.m_camera->m_Film.m_Resolution.GetResY());
  }

  this->resizeGL(m_myVolumeData.m_camera->m_Film.m_Resolution.GetResX(),
                 m_myVolumeData.m_camera->m_Film.m_Resolution.GetResY());

  int MaxSamples = 0;
  glGetIntegerv(GL_MAX_SAMPLES, &MaxSamples);
  LOG_INFO << m_id.toStdString() << " max samples" << MaxSamples;
  glEnable(GL_MULTISAMPLE);

  reset();

  m_rglContext.doneCurrent();
}

void
Renderer::run()
{
  this->init();

  m_rglContext.makeCurrent();

  while (!this->isInterruptionRequested()) {
    this->processRequest();

    // should be harmless... and maybe handle some signal/slot stuff
    QApplication::processEvents();
  }

  m_rglContext.makeCurrent();
  if (m_myVolumeData.ownRenderer) {
    m_myVolumeData.m_renderer->cleanUpResources();
  }
  shutDown();
}

void
Renderer::wakeUp()
{
  m_wait.wakeAll();
}

void
Renderer::addRequest(RenderRequest* request)
{
  m_requestMutex.lock();

  this->m_requests << request;
  this->m_totalQueueDuration += request->getDuration();
  m_requestMutex.unlock();

  this->m_wait.wakeAll();
}

bool
Renderer::processRequest()
{
  m_requestMutex.lock();
  if (m_requests.isEmpty()) {
    m_wait.wait(&m_requestMutex);
  }
  if (m_requests.isEmpty()) {
    m_requestMutex.unlock();
    return false;
  }

  RenderRequest* lastReq = nullptr;
  QImage img;
  if (m_streamMode) {
    QElapsedTimer timer;
    timer.start();

    // eat requests until done, and then render
    // note that any one request could change the streaming mode.
    while (!this->m_requests.isEmpty() && m_streamMode && !this->isInterruptionRequested()) {
      RenderRequest* r = this->m_requests.takeFirst();
      this->m_totalQueueDuration -= r->getDuration();

      std::vector<Command*> cmds = r->getParameters();
      if (cmds.size() > 0) {
        this->processCommandBuffer(r);
      }

      // the true last request will be passed to "emit" and deleted later
      if (!this->m_requests.isEmpty() && m_streamMode) {
        delete r;
        r = nullptr;
        lastReq = nullptr;
      } else {
        lastReq = r;
      }
    }
    if (lastReq) {
      QWebSocket* ws = lastReq->getClient();
      if (ws /* && ws->isValid() && ws->state() == QAbstractSocket::ConnectedState */) {
        LOG_DEBUG << "RENDER for " << ws->peerName().toStdString() << "(" << ws->peerAddress().toString().toStdString()
                  << ":" << QString::number(ws->peerPort()).toStdString() << ")";
      }

      img = this->render();

      lastReq->setActualDuration(timer.nsecsElapsed());

      // in stream mode:
      // if queue is empty, then keep firing redraws back to client, to build up iterations.
      if (m_streamMode) {
        // push another redraw request.
        std::vector<Command*> cmd;
        RequestRedrawCommandD data;
        cmd.push_back(new RequestRedrawCommand(data));
        RenderRequest* rr = new RenderRequest(ws, cmd, false);

        this->m_requests << rr;
        this->m_totalQueueDuration += rr->getDuration();
      }
    }

  } else {
    // if not in stream mode, then process one request, then re-render.
    if (!this->m_requests.isEmpty() && !this->isInterruptionRequested()) {

      // remove request from the queue
      RenderRequest* r = this->m_requests.takeFirst();
      this->m_totalQueueDuration -= r->getDuration();

      // process it
      QElapsedTimer timer;
      timer.start();

      std::vector<Command*> cmds = r->getParameters();
      if (cmds.size() > 0) {
        this->processCommandBuffer(r);
      }

      img = this->render();

      r->setActualDuration(timer.nsecsElapsed());
      lastReq = r;
    }
  }

  // unlock mutex BEFORE emit, in case the signal handler wants to add a request
  m_requestMutex.unlock();

  // inform the server that we are done with r
  // TODO : have a mode where we don't need a QImage
  // and can just return rgba byte array as a thread safe shared ptr
  // writable by render thread and readable by anyone else
  if (!this->isInterruptionRequested()) {
    // TODO look into this way of having the main thread handle this.
    // QMetaObject::invokeMethod(
    //  renderDialog, [=]() { /* ... onRenderRequestProcessed(lastReq, img); ... */ }, Qt::QueuedConnection);
    emit requestProcessed(lastReq, img);
  }
  return true;
}

void
Renderer::processCommandBuffer(RenderRequest* rr)
{
  m_rglContext.makeCurrent();

  std::vector<Command*> cmds = rr->getParameters();
  if (cmds.size() > 0) {
    m_ec.m_renderSettings = &m_myVolumeData.m_renderer->renderSettings();
    m_ec.m_renderer = this;
    m_ec.m_appScene = m_myVolumeData.m_renderer->scene();
    m_ec.m_camera = m_myVolumeData.m_camera;
    m_ec.m_message = "";

    for (auto i = cmds.begin(); i != cmds.end(); ++i) {
      (*i)->execute(&m_ec);
      // commands can fill in the message field of the ec, and we will send it back to the client
      if (!m_ec.m_message.empty()) {
        emit sendString(rr, QString::fromStdString(m_ec.m_message));
        m_ec.m_message = "";
      }
    }
  }
}

QImage
Renderer::render()
{
  QMutexLocker locker(m_openGLMutex);

  m_rglContext.makeCurrent();

  // DRAW
  m_myVolumeData.m_camera->Update();

  if (!m_myVolumeData.m_gesture.graphics.font.get()) {
    m_myVolumeData.m_gesture.graphics.font.reset(new Font());
    std::string fontPath = renderlib::assetPath() + "/fonts/Arial.ttf";
    m_myVolumeData.m_gesture.graphics.font->load(fontPath.c_str());
  }

  SceneView sceneView;
  sceneView.viewport.region = { { 0, 0 }, { m_fbo->width(), m_fbo->height() } };
  sceneView.camera = *(m_myVolumeData.m_camera);
  sceneView.scene = m_myVolumeData.m_renderer->scene();
  sceneView.renderSettings = m_myVolumeData.m_renderSettings;

  // fill gesture graphics with draw commands
  ScaleBarTool scalebar;
  scalebar.clear();
  scalebar.draw(sceneView, m_myVolumeData.m_gesture);

  // main scene rendering
  m_myVolumeData.m_renderer->renderTo(sceneView.camera, m_fbo);

  m_fbo->bind();
  m_myVolumeData.m_gesture.graphics.draw(sceneView, nullptr);
  m_fbo->release();

  std::unique_ptr<uint8_t> bytes(new uint8_t[m_fbo->width() * m_fbo->height() * 4]);
  m_fbo->toImage(bytes.get());
  QImage img = QImage(bytes.get(), m_fbo->width(), m_fbo->height(), QImage::Format_ARGB32).copy().mirrored();

  m_rglContext.doneCurrent();

  return img;
}

void
Renderer::resizeGL(int width, int height)
{
  if ((width == m_width) && (height == m_height)) {
    return;
  }

  QMutexLocker locker(m_openGLMutex);
  m_rglContext.makeCurrent();

  // RESIZE THE RENDER INTERFACE
  if (m_myVolumeData.m_renderer) {
    m_myVolumeData.m_renderer->resize(width, height);
  }

  delete this->m_fbo;
  this->m_fbo = new GLFramebufferObject(width, height, GL_RGBA8);

  glViewport(0, 0, width, height);

  m_width = width;
  m_height = height;
}

void
Renderer::reset(int from)
{
  QMutexLocker locker(m_openGLMutex);

  m_rglContext.makeCurrent();

  glClearColor(0.0, 0.0, 0.0, 1.0);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  // glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE);
  glEnable(GL_BLEND);
  glEnable(GL_LINE_SMOOTH);

  this->m_time.start();

  m_rglContext.doneCurrent();
}

int
Renderer::getTime()
{
  return this->m_time.elapsed();
}

void
Renderer::shutDown()
{
  m_rglContext.makeCurrent();

  delete this->m_fbo;

  delete m_myVolumeData.m_renderSettings;
  m_myVolumeData.m_renderSettings = nullptr;

  delete m_myVolumeData.m_camera;
  m_myVolumeData.m_camera = nullptr;

  delete m_myVolumeData.m_scene;
  m_myVolumeData.m_scene = nullptr;

  if (m_myVolumeData.ownRenderer) {
    delete m_myVolumeData.m_renderer;
  }
  m_myVolumeData.m_renderer = nullptr;

  m_rglContext.doneCurrent();

  m_rglContext.destroy();

  // Stop event processing, move the thread to GUI and make sure it is deleted.
  exit();
  moveToThread(QGuiApplication::instance()->thread());
}
