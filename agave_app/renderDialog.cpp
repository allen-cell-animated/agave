#include "renderDialog.h"
#include "renderer.h"

#include "renderlib/Logging.h"
#include "renderlib/command.h"

#include <QHBoxLayout>
#include <QPainter>
#include <QPushButton>
#include <QVBoxLayout>
#include <QWidget>

ImageDisplay::ImageDisplay(QWidget* parent)
  : QWidget(parent)
{
  m_image = 0;
  setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
  // TODO this image should not be owned by the widget
  m_image = new QImage(256, 256, QImage::Format_RGB888);
  m_image->fill(Qt::white);
}

ImageDisplay::~ImageDisplay()
{
  delete m_image;
}

void
ImageDisplay::setImage(QImage* image)
{
  delete m_image;
  m_image = image;
  repaint();
}

void
ImageDisplay::paintEvent(QPaintEvent*)
{
  QPainter painter(this);
  painter.drawRect(0, 0, width(), height());
  painter.fillRect(0, 0, width(), height(), Qt::black);
  if (!m_image) {
    return;
  }
  // fit image aspect ratio within the given widget rectangle.
  int w = m_image->width();
  int h = m_image->height();
  float imageaspect = (float)w / (float)h;
  float widgetaspect = (float)width() / (float)height();
  QRect targetRect = rect();
  if (imageaspect > widgetaspect) {
    targetRect.setHeight(targetRect.width() / imageaspect);
    targetRect.moveTop((height() - targetRect.height()) / 2);
  } else {
    targetRect.setWidth(targetRect.height() * imageaspect);
    targetRect.moveLeft((width() - targetRect.width()) / 2);
  }
  painter.drawImage(targetRect, *m_image, m_image->rect());
}

RenderDialog::RenderDialog(IRenderWindow* borrowedRenderer,
                           const RenderSettings& renderSettings,
                           const Scene& scene,
                           CCamera camera,
                           QOpenGLContext* glContext,
                           QWidget* parent)
  : m_renderer(borrowedRenderer)
  , m_renderSettings(renderSettings)
  , m_scene(scene)
  , m_camera(camera)
  , m_glContext(glContext)
  , QDialog(parent)
{
  setWindowTitle(tr("Render"));

  mImageView = new ImageDisplay(this);
  mRenderButton = new QPushButton("&Render", this);
  mCloseButton = new QPushButton("&Close", this);

  connect(mRenderButton, SIGNAL(clicked()), this, SLOT(render()));
  connect(mCloseButton, SIGNAL(clicked()), this, SLOT(close()));

  QHBoxLayout* bottomButtonslayout = new QHBoxLayout();
  bottomButtonslayout->addWidget(mRenderButton);
  bottomButtonslayout->addWidget(mCloseButton);

  QVBoxLayout* layout = new QVBoxLayout(this);

  layout->addWidget(mImageView);
  layout->addLayout(bottomButtonslayout);

  setLayout(layout);
}

void
RenderDialog::setImage(QImage* image)
{
  mImageView->setImage(image);
}

void
RenderDialog::render()
{
  LOG_INFO << "Render button clicked";
  // when render is done, draw QImage to widget and save to file if autosave?
  Renderer* r = new Renderer("Render dialog render thread ", this, m_mutex);
  // now get our rendering resources into this Renderer object
  r->configure(m_renderer, m_renderSettings, m_scene, m_camera, m_glContext);
  m_glContext->moveToThread(r);
  r->setStreamMode(1);

  std::vector<Command*> cmd;
  RequestRedrawCommandD data;
  cmd.push_back(new RequestRedrawCommand(data));
  r->addRequest(new RenderRequest(nullptr, cmd, false));

  this->m_renderThread = r;
  // queued across thread boundary.  typically requestProcessed is called from another thread.
  // BlockingQueuedConnection forces send to happen immediately after render.  Default (QueuedConnection) will be fully
  // async.
  static int N = 1;
  connect(
    r,
    &Renderer::requestProcessed,
    this,
    [this](RenderRequest* req, QImage image) {
      this->setImage(new QImage(image));
      N = N + 1;
      if (N % 100 == 0) {
        image.save("~/test.png"); // "C:\\Users\\dmt\\test.png");
      }
    },
    Qt::BlockingQueuedConnection);
  // connect(r, SIGNAL(sendString(RenderRequest*, QString)), this, SLOT(sendString(RenderRequest*, QString)));
  LOG_INFO << "Starting render thread...";
  r->start();
}

void
RenderDialog::done(int r)
{
  this->m_renderer = nullptr;
  this->m_renderThread->requestInterruption();
  bool ok = this->m_renderThread->wait(QDeadlineTimer(2000));
  if (ok) {
    LOG_DEBUG << "Render thread stopped cleanly";
  } else {
    LOG_DEBUG << "Render thread did not stop cleanly";
  }
  m_renderThread->deleteLater();
  QDialog::done(r);
}
