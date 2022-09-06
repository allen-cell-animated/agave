#include "renderDialog.h"
#include "renderer.h"

#include "renderlib/Logging.h"
#include "renderlib/RenderGLPT.h"
#include "renderlib/command.h"

#include <QApplication>
#include <QComboBox>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QImageWriter>
#include <QLabel>
#include <QLineEdit>
#include <QPainter>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QTimeEdit>
#include <QVBoxLayout>
#include <QWidget>

ImageDisplay::ImageDisplay(QWidget* parent)
  : QWidget(parent)
{
  m_pixmap = nullptr;
  m_image = nullptr;
  setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
  // TODO this image should not be owned by the widget
  m_image = new QImage(256, 256, QImage::Format_RGB888);
  m_image->fill(Qt::white);
  m_pixmap = new QPixmap(256, 256);
  m_pixmap->convertFromImage(*m_image);
}

ImageDisplay::~ImageDisplay()
{
  delete m_image;
  delete m_pixmap;
}

void
ImageDisplay::setImage(QImage* image)
{
  delete m_image;
  m_image = image;
  if (image) {
    m_pixmap->convertFromImage(*m_image);
  }
  repaint();
}

void
ImageDisplay::save(QString filename)
{
  // m_pixmap->save(filename);
  m_image->save(filename);
}

void
ImageDisplay::paintEvent(QPaintEvent*)
{
  QPainter painter(this);
  painter.drawRect(0, 0, width(), height());
  painter.fillRect(0, 0, width(), height(), Qt::darkGray);
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
  // painter.drawImage(targetRect, *m_image, m_image->rect());
  painter.drawPixmap(targetRect, *m_pixmap, m_pixmap->rect());
}

struct ResolutionPreset
{
  const char* label;
  int w;
  int h;
};
ResolutionPreset resolutionPresets[] = {
  { "4:3 640x480", 640, 480 },       { "4:3 1024x768", 1024, 768 },     { "4:3 1280x960", 1280, 960 },
  { "4:3 1600x1200", 1600, 1200 },   { "16:9 640x360", 640, 360 },      { "16:9 960x540", 960, 540 },
  { "16:9 1280x720", 1280, 720 },    { "16:9 1920x1080", 1920, 1080 },  { "16:9 3840x2160", 3840, 2160 },
  { "16:10 1920x1200", 1920, 1200 }, { "16:10 2560x1440", 2560, 1440 }, { "16:10 3840x2160", 3840, 2160 },
};

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
  , m_renderThread(nullptr)
  , m_totalRenderTime(0)
  , mWidth(0)
  , mHeight(0)
  , QDialog(parent)
{
  setWindowTitle(tr("AGAVE Render"));

  mImageView = new ImageDisplay(this);
  mRenderButton = new QPushButton("&Render", this);
  mPauseRenderButton = new QPushButton("&Pause", this);
  mStopRenderButton = new QPushButton("&Stop", this);
  mSaveButton = new QPushButton("&Save", this);
  mCloseButton = new QPushButton("&Close", this);
  static const int defaultRenderIterations = 1024;
  mProgressBar = new QProgressBar(this);
  mProgressBar->setRange(0, defaultRenderIterations);

  mRenderDurationEdit = new QComboBox(this);
  mRenderDurationEdit->addItem("Samples");
  mRenderDurationEdit->addItem("Time");

  mRenderSamplesEdit = new QSpinBox(this);
  mRenderSamplesEdit->setMinimum(1);
  mRenderSamplesEdit->setMaximum(65536);
  mRenderSamplesEdit->setValue(defaultRenderIterations);
  mRenderTimeEdit = new QTimeEdit(this);
  mRenderTimeEdit->setDisplayFormat("hh:mm:ss");
  mRenderTimeEdit->setTime(QTime(0, 0, 30));

  setRenderDurationType(eRenderDurationType::SAMPLES);

  mWidth = camera.m_Film.m_Resolution.GetResX();
  mHeight = camera.m_Film.m_Resolution.GetResY();

  mWidthInput = new QSpinBox(this);
  mWidthInput->setMaximum(4096);
  mWidthInput->setMinimum(2);
  mWidthInput->setValue(mWidth);
  mHeightInput = new QSpinBox(this);
  mHeightInput->setMaximum(4096);
  mHeightInput->setMinimum(2);
  mHeightInput->setValue(mHeight);
  mResolutionPresets = new QComboBox(this);
  mResolutionPresets->addItem("Choose a preset...");
  for (int i = 0; i < sizeof(resolutionPresets) / sizeof(ResolutionPreset); i++) {
    mResolutionPresets->addItem(resolutionPresets[i].label);
  }

  connect(mRenderButton, SIGNAL(clicked()), this, SLOT(render()));
  connect(mPauseRenderButton, SIGNAL(clicked()), this, SLOT(pauseRendering()));
  connect(mStopRenderButton, SIGNAL(clicked()), this, SLOT(stopRendering()));
  connect(mSaveButton, SIGNAL(clicked()), this, SLOT(save()));
  connect(mCloseButton, SIGNAL(clicked()), this, SLOT(close()));
  connect(mResolutionPresets, SIGNAL(currentIndexChanged(int)), this, SLOT(onResolutionPreset(int)));
  connect(mWidthInput, SIGNAL(valueChanged(int)), this, SLOT(updateWidth(int)));
  connect(mHeightInput, SIGNAL(valueChanged(int)), this, SLOT(updateHeight(int)));
  connect(mRenderSamplesEdit, SIGNAL(valueChanged(int)), this, SLOT(updateRenderSamples(int)));
  connect(mRenderTimeEdit, SIGNAL(valueChanged(QTime)), this, SLOT(updateRenderTime(QTime)));
  connect(mRenderDurationEdit, SIGNAL(currentIndexChanged(int)), this, SLOT(onRenderDurationTypeChanged(int)));

  QHBoxLayout* topButtonsLayout = new QHBoxLayout();
  topButtonsLayout->addWidget(new QLabel(tr("X:")), 0);
  topButtonsLayout->addWidget(mWidthInput, 1);
  topButtonsLayout->addWidget(new QLabel(tr("Y:")), 0);
  topButtonsLayout->addWidget(mHeightInput, 1);
  topButtonsLayout->addWidget(mResolutionPresets, 1);

  QHBoxLayout* durationsLayout = new QHBoxLayout();
  durationsLayout->addWidget(mRenderDurationEdit, 0);
  durationsLayout->addWidget(new QLabel(tr("Time:")), 0);
  durationsLayout->addWidget(mRenderTimeEdit, 1);
  durationsLayout->addWidget(new QLabel(tr("Samples:")), 0);
  durationsLayout->addWidget(mRenderSamplesEdit, 1);
  //  durationsLayout->addWidget(mRenderDurationTypeEdit, 1);

  QHBoxLayout* bottomButtonslayout = new QHBoxLayout();
  bottomButtonslayout->addWidget(mRenderButton);
  bottomButtonslayout->addWidget(mPauseRenderButton);
  bottomButtonslayout->addWidget(mStopRenderButton);
  bottomButtonslayout->addWidget(mSaveButton);
  bottomButtonslayout->addWidget(mCloseButton);

  QVBoxLayout* layout = new QVBoxLayout(this);

  layout->addLayout(topButtonsLayout);
  layout->addLayout(durationsLayout);
  layout->addWidget(mImageView);
  layout->addWidget(mProgressBar);
  layout->addLayout(bottomButtonslayout);

  setLayout(layout);
}

void
RenderDialog::save()
{
  // pause rendering
  pauseRendering();

  QFileDialog::Options options;
#ifdef __linux__
  options |= QFileDialog::DontUseNativeDialog;
#endif

  const QByteArrayList supportedFormats = QImageWriter::supportedImageFormats();
  QStringList supportedFormatStrings;
  foreach (const QByteArray& item, supportedFormats) {
    supportedFormatStrings.append(QString::fromLocal8Bit(item)); // Assuming local 8-bit.
  }

  static const QStringList desiredFormats = { "png", "jpg", "tif" };

  QStringList formatFilters;
  foreach (const QString& desiredFormatName, desiredFormats) {
    if (supportedFormatStrings.contains(desiredFormatName, Qt::CaseInsensitive)) {
      formatFilters.append(desiredFormatName.toUpper() + " (*." + desiredFormatName + ")");
    }
  }
  QString allSupportedFormatsFilter = formatFilters.join(";;");

  QString fileName =
    QFileDialog::getSaveFileName(this, tr("Save Image"), QString(), allSupportedFormatsFilter, nullptr, options);

  if (fileName.isEmpty()) {
    return;
  }
  mImageView->save(fileName);
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
  if (!this->m_renderThread) {

    // when render is done, draw QImage to widget and save to file if autosave?
    Renderer* r = new Renderer("Render dialog render thread ", this, m_mutex);
    this->m_renderThread = r;
    // now get our rendering resources into this Renderer object
    r->configure(m_renderer, m_renderSettings, m_scene, m_camera, m_glContext);
    m_glContext->moveToThread(r);

    // first time in, set up stream mode and give the first draw request
    resumeRendering();

    // queued across thread boundary.  typically requestProcessed is called from another thread.
    // BlockingQueuedConnection forces send to happen immediately after render.  Default (QueuedConnection) will be
    // fully async.
    connect(
      r,
      &Renderer::requestProcessed,
      this,
      [this](RenderRequest* req, QImage image) {
        this->m_totalRenderTime += req->getActualDuration();
        // increment progress
        if (mRenderDurationType == eRenderDurationType::SAMPLES) {
          mProgressBar->setValue(mProgressBar->value() + 1);
        } else {
          // nano to seconds.  render durations in the dialog are specified in seconds.
          mProgressBar->setValue(m_totalRenderTime / (1000 * 1000 * 1000));
        }
        if (mProgressBar->value() >= mProgressBar->maximum()) {
          stopRendering();
        } else {
          // update display
          this->setImage(new QImage(image));
        }
      },
      Qt::BlockingQueuedConnection);
    // connect(r, SIGNAL(sendString(RenderRequest*, QString)), this, SLOT(sendString(RenderRequest*, QString)));
    LOG_INFO << "Starting render thread...";
    r->start();
  } else {
    resumeRendering();
  }
}

void
RenderDialog::pauseRendering()
{
  if (m_renderThread) {

    m_renderThread->setStreamMode(0);
  }
}

void
RenderDialog::stopRendering()
{
  if (m_renderThread && m_renderThread->isRunning()) {
    m_renderThread->setStreamMode(0);

    resetProgress();
  }
}

void
RenderDialog::endRenderThread()
{
  pauseRendering();
  if (m_renderThread && m_renderThread->isRunning()) {
    m_renderThread->requestInterruption();
    m_renderThread->wakeUp();

    bool ok = false;
    int n = 0;
    while (!ok && n < 30) {
      ok = m_renderThread->wait(QDeadlineTimer(20));
      n = n + 1;
      QApplication::processEvents();
    }
    if (ok) {
      LOG_DEBUG << "Render thread stopped cleanly after " << n << " tries";
    } else {
      LOG_DEBUG << "Render thread did not stop cleanly";
    }
  }
}

void
RenderDialog::resumeRendering()
{
  if (m_renderThread) {
    m_renderThread->setStreamMode(1);

    std::vector<Command*> cmd;
    SetResolutionCommandD resdata;
    resdata.m_x = mWidth;
    resdata.m_y = mHeight;
    cmd.push_back(new SetResolutionCommand(resdata));
    RequestRedrawCommandD data;
    cmd.push_back(new RequestRedrawCommand(data));
    m_renderThread->addRequest(new RenderRequest(nullptr, cmd, false));
  }
}

void
RenderDialog::done(int r)
{
  this->m_renderer = nullptr;
  endRenderThread();
  if (m_renderThread) {
    m_renderThread->deleteLater();
  }
  QDialog::done(r);
}

void
RenderDialog::onResolutionPreset(int index)
{
  // find preset res and set w/h
  const ResolutionPreset& preset = resolutionPresets[index - 1];
  mWidthInput->setValue((preset.w));
  mHeightInput->setValue((preset.h));
  emit setRenderResolution(mWidth, mHeight);
}

void
RenderDialog::updateWidth(int w)
{
  mWidth = w;
  m_camera.m_Film.m_Resolution.SetResX(w);
  resetProgress();

  emit setRenderResolution(mWidth, mHeight);
}

void
RenderDialog::updateHeight(int h)
{
  mHeight = h;
  m_camera.m_Film.m_Resolution.SetResY(h);
  resetProgress();
  emit setRenderResolution(mWidth, mHeight);
}

int
RenderDialog::getXResolution()
{
  return m_camera.m_Film.m_Resolution.GetResX();
}

int
RenderDialog::getYResolution()
{
  return m_camera.m_Film.m_Resolution.GetResY();
}

void
RenderDialog::setRenderDurationType(eRenderDurationType type)
{
  mRenderDurationType = type;
  if (mRenderDurationType == eRenderDurationType::SAMPLES) {
    mRenderTimeEdit->setEnabled(false);
    mRenderSamplesEdit->setEnabled(true);
  } else {
    mRenderTimeEdit->setEnabled(true);
    mRenderSamplesEdit->setEnabled(false);
  }
}

void
RenderDialog::updateRenderSamples(int s)
{
  if (mRenderDurationType == eRenderDurationType::SAMPLES) {
    mProgressBar->setMaximum(s);
  }
}

void
RenderDialog::updateRenderTime(QTime t)
{
  if (mRenderDurationType == eRenderDurationType::TIME) {
    mProgressBar->setMaximum(t.hour() * 3600 + t.minute() * 60 + t.second());
  }
}

void
RenderDialog::resetProgress()
{
  RenderGLPT* r = dynamic_cast<RenderGLPT*>(m_renderer);
  r->getRenderSettings().SetNoIterations(0);

  mProgressBar->reset();
  m_totalRenderTime = 0;
}

void
RenderDialog::onRenderDurationTypeChanged(int index)
{
  if (index == 0) {
    setRenderDurationType(eRenderDurationType::SAMPLES);
    updateRenderSamples(mRenderSamplesEdit->value());
  } else {
    setRenderDurationType(eRenderDurationType::TIME);
    updateRenderTime(mRenderTimeEdit->time());
  }
}