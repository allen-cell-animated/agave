#include "pyrenderer.h"

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

OffscreenRenderer::OffscreenRenderer()
  : fbo(nullptr)
  , _width(0)
  , _height(0)
{
  LOG_DEBUG << "Initializing renderer for python script";
  this->init();
}

OffscreenRenderer::~OffscreenRenderer()
{
  this->context->makeCurrent(this->surface);
  myVolumeData._renderer->cleanUpResources();
  shutDown();
}

void
OffscreenRenderer::myVolumeInit()
{
  myVolumeData._renderSettings = new RenderSettings();

  myVolumeData._camera = new CCamera();
  myVolumeData._camera->m_Film.m_ExposureIterations = 1;

  myVolumeData._scene = new Scene();
  myVolumeData._scene->initLights();

  myVolumeData._renderer = new RenderGLPT(myVolumeData._renderSettings);
  myVolumeData._renderer->initialize(1024, 1024);
  myVolumeData._renderer->setScene(myVolumeData._scene);
}

void
OffscreenRenderer::init()
{
  myVolumeInit();

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

  // int status = gladLoadGL();
  // if (!status) {
  //  LOG_INFO << "COULD NOT LOAD GL";
  //}

  ///////////////////////////////////
  // INIT THE RENDER LIB
  ///////////////////////////////////

  this->resizeGL(1024, 1024);

  int MaxSamples = 0;
  glGetIntegerv(GL_MAX_SAMPLES, &MaxSamples);
  LOG_INFO << "max samples" << MaxSamples;

  glEnable(GL_MULTISAMPLE);

  reset();

  this->context->doneCurrent();
}

#if 0 
void
OffscreenRenderer::processCommandBuffer(RenderRequest* rr)
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
#endif

QImage
OffscreenRenderer::render()
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

void
OffscreenRenderer::resizeGL(int width, int height)
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

void
OffscreenRenderer::reset(int from)
{
  this->context->makeCurrent(this->surface);

  glClearColor(0.0, 0.0, 0.0, 1.0);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  // glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ONE);
  glEnable(GL_BLEND);
  glEnable(GL_LINE_SMOOTH);
}

void
OffscreenRenderer::shutDown()
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
}
