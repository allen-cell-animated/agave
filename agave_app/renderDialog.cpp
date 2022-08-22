#include "renderDialog.h"
#include "renderer.h"

#include "renderlib/Logging.h"

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

void
ImageDisplay::setImage(QImage* image)
{
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
                           QWidget* parent)
  : m_renderer(borrowedRenderer)
  , m_renderSettings(renderSettings)
  , m_scene(scene)
  , m_camera(camera)
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
  QMutex mutex;
  Renderer* r = new Renderer("Thread " + QString::number(i), this, mutex);
  this->m_renderThread = r;
  // queued across thread boundary.  typically requestProcessed is called from another thread.
  // BlockingQueuedConnection forces send to happen immediately after render.  Default (QueuedConnection) will be fully
  // async.
  connect(
    r,
    SIGNAL(requestProcessed(RenderRequest*, QImage)),
    this,
    [this](RenderRequest* req, QImage image) { this->setImage(image); },
    Qt::BlockingQueuedConnection);
  // connect(r, SIGNAL(sendString(RenderRequest*, QString)), this, SLOT(sendString(RenderRequest*, QString)));
  LOG_INFO << "Starting render thread...";
  r->start();
}

void
RenderDialog::stop()
{
  this->m_renderer = nullptr;
  this->m_renderThread->stop();
  delete m_renderThread;
}
