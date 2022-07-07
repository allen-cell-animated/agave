#include "renderDialog.h"

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

RenderDialog::RenderDialog(QWidget* parent)
  : QDialog(parent)
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
RenderDialog::render()
{
  LOG_INFO << "Render button clicked";
  // when render is done, draw QImage to widget and save to file if autosave?
}