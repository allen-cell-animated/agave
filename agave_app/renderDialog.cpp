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
#include <QMouseEvent>
#include <QPainter>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QStandardPaths>
#include <QTimeEdit>
#include <QVBoxLayout>
#include <QWidget>

static const float ZOOM_STEP = 1.1f;

ImageDisplay::ImageDisplay(QWidget* parent)
  : QWidget(parent)
  , m_scale(1.0)
{
  m_pixmap = nullptr;
  setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
  // TODO this image should not be owned by the widget
  // m_image = new QImage(256, 256, QImage::Format_RGB888);
  // m_image->fill(Qt::white);
  m_pixmap = new QPixmap(256, 256);
  //  m_pixmap->convertFromImage(*m_image);
}

ImageDisplay::~ImageDisplay()
{
  delete m_pixmap;
}

void
ImageDisplay::setImage(QImage* image)
{
  if (image) {
    // TODO: sometimes we want to hold on to the transparent image with alphas, sometimes we don't!!!!
    // users may want to save the image with or without the background!
    // can this step be skipped at all?
    QImage im = image->convertToFormat(QImage::Format_RGB32);

    m_pixmap->convertFromImage(im, Qt::ColorOnly);
    m_rect = m_pixmap->rect();
    m_rect.translate(-m_rect.center());
  }
  repaint(); // update() ?
}

void
ImageDisplay::save(QString filename)
{
  m_pixmap->save(filename);
}

void
ImageDisplay::paintEvent(QPaintEvent*)
{
  QPainter painter(this);
  painter.setRenderHint(QPainter::SmoothPixmapTransform);

  painter.drawRect(0, 0, width(), height());
  painter.fillRect(0, 0, width(), height(), Qt::darkGray);

  if (!m_pixmap) {
    return;
  }
  painter.translate(rect().center());
  painter.scale(m_scale, m_scale);
  painter.translate(m_delta);
  painter.drawPixmap(m_rect.topLeft(), *m_pixmap);
}

void
ImageDisplay::mousePressEvent(QMouseEvent* event)
{
  m_reference = event->pos();
  qApp->setOverrideCursor(Qt::ClosedHandCursor);
  setMouseTracking(true);
}

void
ImageDisplay::mouseMoveEvent(QMouseEvent* event)
{
  m_delta += (event->pos() - m_reference) * 1.0 / m_scale;
  m_reference = event->pos();
  update();
}

void
ImageDisplay::mouseReleaseEvent(QMouseEvent*)
{
  qApp->restoreOverrideCursor();
  setMouseTracking(false);
}

void
ImageDisplay::scale(qreal s)
{
  m_scale *= s;
  repaint();
  // update();
}
void
ImageDisplay::setScale(qreal s)
{
  m_scale = s;
  repaint();
  // update();
}

void
ImageDisplay::fit(int w, int h)
{
  // fit image aspect ratio within the given widget rectangle.
  float imageaspect = (float)w / (float)h;
  float widgetaspect = (float)width() / (float)height();
  // targetRect will describe a sub-rectangle of the ImageDisplay's client rect
  QRect targetRect = rect();
  if (imageaspect > widgetaspect) {
    targetRect.setHeight(targetRect.width() / imageaspect);
    targetRect.moveTop((height() - targetRect.height()) / 2);
    // scale value from width!
    m_scale = ((float)targetRect.width() / (float)w);
  } else {
    targetRect.setWidth(targetRect.height() * imageaspect);
    targetRect.moveLeft((width() - targetRect.width()) / 2);
    // scale value from height!
    m_scale = ((float)targetRect.height() / (float)h);
  }

  // center image
  m_delta = QPointF();
  m_reference = QPointF();

  repaint();
  // update();
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
                           std::string volumeFilePath,
                           CaptureSettings* captureSettings,
                           QWidget* parent)
  : m_renderer(borrowedRenderer)
  , m_renderSettings(renderSettings)
  , m_scene(scene)
  , m_camera(camera)
  , m_glContext(glContext)
  , mVolumeFilePath(volumeFilePath)
  , m_renderThread(nullptr)
  , m_frameRenderTime(0)
  , mWidth(0)
  , mHeight(0)
  , mFrameNumber(0)
  , mTotalFrames(1)
  , mCaptureSettings(captureSettings)
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

  mFrameProgressBar = new QProgressBar(this);
  if (mCaptureSettings->durationType == eRenderDurationType::SAMPLES) {
    mFrameProgressBar->setRange(0, mCaptureSettings->samples);
  } else {
    mFrameProgressBar->setRange(0, mCaptureSettings->duration);
  }

  mRenderDurationEdit = new QComboBox(this);
  mRenderDurationEdit->addItem("Samples", eRenderDurationType::SAMPLES);
  mRenderDurationEdit->addItem("Time", eRenderDurationType::TIME);
  auto mapDurationTypeToUIIndex = std::map<eRenderDurationType, int>{
    { eRenderDurationType::SAMPLES, 0 },
    { eRenderDurationType::TIME, 1 },
  };
  mRenderDurationEdit->setCurrentIndex(mapDurationTypeToUIIndex[mCaptureSettings->durationType]);

  mRenderSamplesEdit = new QSpinBox(this);
  mRenderSamplesEdit->setMinimum(1);
  mRenderSamplesEdit->setMaximum(65536);
  mRenderSamplesEdit->setValue(mCaptureSettings->samples);
  mRenderTimeEdit = new QTimeEdit(this);
  mRenderTimeEdit->setDisplayFormat("hh:mm:ss");
  int h = mCaptureSettings->duration / (60 * 60);
  int m = (mCaptureSettings->duration - h * 60 * 60) / 60;
  int s = (mCaptureSettings->duration - h * 60 * 60 - m * 60);
  mRenderTimeEdit->setTime(QTime(h, m, s));

  setRenderDurationType(mCaptureSettings->durationType);

  mWidth = mCaptureSettings->width;
  mHeight = mCaptureSettings->height;
  // copy the window width in case user wants this
  // mWidth = camera.m_Film.m_Resolution.GetResX();
  // mHeight = camera.m_Film.m_Resolution.GetResY();

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

  mZoomInButton = new QPushButton("+", this);
  mZoomOutButton = new QPushButton("-", this);
  mZoomFitButton = new QPushButton("[]", this);

  mStartTimeInput = new QSpinBox(this);
  mStartTimeInput->setMinimum(0);
  mStartTimeInput->setMaximum(scene.m_timeLine.maxTime());
  mStartTimeInput->setValue(mCaptureSettings->startTime);
  mEndTimeInput = new QSpinBox(this);
  mEndTimeInput->setMinimum(0);
  mEndTimeInput->setMaximum(scene.m_timeLine.maxTime());
  mEndTimeInput->setValue(mCaptureSettings->endTime);

  mTimeSeriesProgressBar = new QProgressBar(this);
  mTimeSeriesProgressBar->setRange(0, abs(mEndTimeInput->value() - mStartTimeInput->value()) + 1);

  mSelectSaveDirectoryButton = new QPushButton("Dir...", this);

  mSaveDirectoryLabel = new QLabel(QString::fromStdString(mCaptureSettings->outputDir), this);
  mSaveFilePrefix = new QLineEdit(QString::fromStdString(mCaptureSettings->filenamePrefix), this);

  connect(mRenderButton, &QPushButton::clicked, this, &RenderDialog::render);
  connect(mPauseRenderButton, &QPushButton::clicked, this, &RenderDialog::pauseRendering);
  connect(mStopRenderButton, &QPushButton::clicked, this, &RenderDialog::stopRendering);
  connect(mSaveButton, &QPushButton::clicked, this, &RenderDialog::save);
  connect(mCloseButton, &QPushButton::clicked, this, &RenderDialog::close);
  connect(mResolutionPresets, SIGNAL(currentIndexChanged(int)), this, SLOT(onResolutionPreset(int)));
  connect(mWidthInput, SIGNAL(valueChanged(int)), this, SLOT(updateWidth(int)));
  connect(mHeightInput, SIGNAL(valueChanged(int)), this, SLOT(updateHeight(int)));
  connect(mRenderSamplesEdit, SIGNAL(valueChanged(int)), this, SLOT(updateRenderSamples(int)));
  connect(mRenderTimeEdit, SIGNAL(timeChanged(const QTime&)), this, SLOT(updateRenderTime(const QTime&)));
  connect(mRenderDurationEdit, SIGNAL(currentIndexChanged(int)), this, SLOT(onRenderDurationTypeChanged(int)));
  connect(mZoomInButton, &QPushButton::clicked, this, &RenderDialog::onZoomInClicked);
  connect(mZoomOutButton, &QPushButton::clicked, this, &RenderDialog::onZoomOutClicked);
  connect(mZoomFitButton, &QPushButton::clicked, this, &RenderDialog::onZoomFitClicked);
  connect(mStartTimeInput, QOverload<int>::of(&QSpinBox::valueChanged), this, &RenderDialog::onStartTimeChanged);
  connect(mEndTimeInput, QOverload<int>::of(&QSpinBox::valueChanged), this, &RenderDialog::onEndTimeChanged);
  connect(mSelectSaveDirectoryButton, &QPushButton::clicked, this, &RenderDialog::onSelectSaveDirectoryClicked);
  connect(mSaveFilePrefix, &QLineEdit::textChanged, this, &RenderDialog::onSaveFilePrefixChanged);

  QHBoxLayout* topButtonsLayout = new QHBoxLayout();
  topButtonsLayout->addWidget(new QLabel(tr("X:")), 0);
  topButtonsLayout->addWidget(mWidthInput, 1);
  topButtonsLayout->addWidget(new QLabel(tr("Y:")), 0);
  topButtonsLayout->addWidget(mHeightInput, 1);
  topButtonsLayout->addWidget(mResolutionPresets, 1);
  topButtonsLayout->addWidget(new QLabel(tr("T0:")), 0);
  topButtonsLayout->addWidget(mStartTimeInput, 1);
  topButtonsLayout->addWidget(new QLabel(tr("T1:")), 0);
  topButtonsLayout->addWidget(mEndTimeInput, 1);

  QHBoxLayout* saveButtonsLayout = new QHBoxLayout();
  saveButtonsLayout->addWidget(mSaveFilePrefix, 1);
  saveButtonsLayout->addWidget(mSaveDirectoryLabel, 1);
  saveButtonsLayout->addWidget(mSelectSaveDirectoryButton, 1);

  QHBoxLayout* zoomButtonsLayout = new QHBoxLayout();
  zoomButtonsLayout->addWidget(mZoomInButton, 1);
  zoomButtonsLayout->addWidget(mZoomOutButton, 1);
  zoomButtonsLayout->addWidget(mZoomFitButton, 1);

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

  // TODO see layout->setmenubar !
  layout->addLayout(topButtonsLayout);
  layout->addLayout(durationsLayout);
  layout->addLayout(saveButtonsLayout);
  layout->addLayout(zoomButtonsLayout);
  layout->addWidget(mImageView);
  layout->addWidget(mFrameProgressBar);
  layout->addWidget(mTimeSeriesProgressBar);
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

    mFrameProgressBar->setValue(0);
    mTimeSeriesProgressBar->setValue(0);
    mFrameNumber = mStartTimeInput->value();

    // when render is done, draw QImage to widget and save to file if autosave?
    Renderer* r = new Renderer("Render dialog render thread ", this, m_mutex);
    this->m_renderThread = r;
    // now get our rendering resources into this Renderer object
    r->configure(m_renderer, m_renderSettings, m_scene, m_camera, mVolumeFilePath, m_glContext);
    m_glContext->moveToThread(r);

    onZoomFitClicked();
    // first time in, set up stream mode and give the first draw request
    resumeRendering();

    // queued across thread boundary.  typically requestProcessed is called from another thread.
    // BlockingQueuedConnection forces send to happen immediately after render.  Default (QueuedConnection) will be
    // fully async.
    connect(
      r, &Renderer::requestProcessed, this, &RenderDialog::onRenderRequestProcessed, Qt::BlockingQueuedConnection);
    // connect(r, SIGNAL(sendString(RenderRequest*, QString)), this, SLOT(sendString(RenderRequest*, QString)));
    LOG_INFO << "Starting render thread...";
    r->start();
  } else {
    resumeRendering();
  }
}

void
RenderDialog::onRenderRequestProcessed(RenderRequest* req, QImage image)
{
  // note that every render request that comes thru here sends a
  // whole new image.
  // this is likely much less efficient than writing the image in-place

  // this is called after the render thread has completed a render request
  // the QImage is sent here from the thread.
  // this is an incremental update of a render and our chance to update the GUI and state of our processing

  static int imagesReceived = 0;
  imagesReceived = imagesReceived + 1;

  if (mTimeSeriesProgressBar->value() >= mTimeSeriesProgressBar->maximum() + 1) {
    LOG_DEBUG << "received frame after timeline completed";
    return;
  }

  // increment our time counter
  this->m_frameRenderTime += req->getActualDuration();

  // increment progress
  if (mRenderDurationType == eRenderDurationType::SAMPLES) {
    mFrameProgressBar->setValue(mFrameProgressBar->value() + 1);
    // mFrameProgressBar->setValue(imagesReceived);
  } else {
    // nano to seconds.  render durations in the dialog are specified in seconds.
    mFrameProgressBar->setValue(m_frameRenderTime / (1000 * 1000 * 1000));
  }

  // did a frame finish?
  if (mFrameProgressBar->value() >= mFrameProgressBar->maximum()) {
    // we just received the last sample.
    // however, another sample was already enqueued!!!!
    // so we know we will have one sample to discard.

    LOG_DEBUG << imagesReceived << " images received";
    LOG_DEBUG << "Progress " << mFrameProgressBar->value() << " / " << mFrameProgressBar->maximum();
    LOG_DEBUG << "frame " << mFrameNumber << " progress completed";
    imagesReceived = 0;
    // update display with finished frame
    this->setImage(&image);

    // save image
    if (true) {
      // TODO set up autosave path when we start rendering
      QString autosavePath = mSaveDirectoryLabel->text();
      QDir d(autosavePath);
      bool pathOk = d.mkpath(autosavePath);
      if (!pathOk) {
        LOG_ERROR << "Failed to make path " << autosavePath.toStdString();
      }
      // save!
      QString filename = mSaveFilePrefix->text() + QString("_%1.png").arg(mFrameNumber, 4, 10, QChar('0'));
      QFileInfo fileInfo(d, filename);
      QString saveFilePath = fileInfo.absoluteFilePath();
      bool ok = image.save(saveFilePath);
      if (!ok) {
        LOG_ERROR << "Failed to save render " << saveFilePath.toStdString();
      } else {
        LOG_INFO << "Saved " << saveFilePath.toStdString();
      }
    }

    // increment frame
    mFrameNumber += 1;
    mTimeSeriesProgressBar->setValue(mTimeSeriesProgressBar->value() + 1);
    LOG_DEBUG << "Total Progress " << mTimeSeriesProgressBar->value() << " / " << mTimeSeriesProgressBar->maximum();

    // done with LAST frame? halt everything.
    if (mTimeSeriesProgressBar->value() >= mTimeSeriesProgressBar->maximum()) {
      LOG_DEBUG << "all frames completed.  pausing render";
      pauseRendering();
      // stopRendering();
    } else {
      LOG_DEBUG << "reset frame progress for next frame";
      // reset progress and render time
      mFrameProgressBar->setValue(0);
      m_frameRenderTime = 0;

      // m_scene.m_timeLine.increment(1);

      // set up for next frame!
      //
      // run some code to increment T or rotation angle etc.
      RenderGLPT* r = dynamic_cast<RenderGLPT*>(m_renderer);
      r->getRenderSettings().SetNoIterations(0);

      LOG_DEBUG << "queue setTime " << mFrameNumber << " command ";
      std::vector<Command*> cmd;
      SetTimeCommandD timedata;
      timedata.m_time = mFrameNumber;
      cmd.push_back(new SetTimeCommand(timedata));
      m_renderThread->addRequest(new RenderRequest(nullptr, cmd, false));
    }

  } else {
    // update display if it's time to do so.
    this->setImage(&image);
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
    SetTimeCommandD timedata;
    timedata.m_time = mFrameNumber;
    cmd.push_back(new SetTimeCommand(timedata));
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
  mCaptureSettings->width = w;
  resetProgress();

  emit setRenderResolution(mWidth, mHeight);
}

void
RenderDialog::updateHeight(int h)
{
  mHeight = h;
  m_camera.m_Film.m_Resolution.SetResY(h);
  mCaptureSettings->height = h;
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
  mCaptureSettings->durationType = type;

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
  mCaptureSettings->samples = s;
  if (mRenderDurationType == eRenderDurationType::SAMPLES) {
    mFrameProgressBar->setMaximum(s);
  }
}

void
RenderDialog::updateRenderTime(const QTime& t)
{
  mCaptureSettings->duration = t.hour() * 60 * 60 + t.minute() * 60 + t.second();

  if (mRenderDurationType == eRenderDurationType::TIME) {
    mFrameProgressBar->setMaximum(mCaptureSettings->duration);
  }
}

void
RenderDialog::resetProgress()
{
  RenderGLPT* r = dynamic_cast<RenderGLPT*>(m_renderer);
  r->getRenderSettings().SetNoIterations(0);

  mFrameProgressBar->reset();
  m_frameRenderTime = 0; // FIX per frame render time vs total render time elapsed

  // TODO reset time series too?
  mTimeSeriesProgressBar->reset();
  mFrameNumber = mStartTimeInput->value();
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

void
RenderDialog::onZoomInClicked()
{
  mImageView->scale(ZOOM_STEP);
}
void
RenderDialog::onZoomOutClicked()
{
  mImageView->scale(1.0 / ZOOM_STEP);
}
void
RenderDialog::onZoomFitClicked()
{
  // fit image aspect ratio within the given widget rectangle.
  int w = mWidth;
  int h = mHeight;
  mImageView->fit(w, h);
}

void
RenderDialog::onStartTimeChanged(int t)
{
  mTimeSeriesProgressBar->setRange(0, abs(mEndTimeInput->value() - mStartTimeInput->value()) + 1);

  mTotalFrames = abs(mEndTimeInput->value() - mStartTimeInput->value()) + 1;

  mCaptureSettings->startTime = t;
}
void
RenderDialog::onEndTimeChanged(int t)
{
  mTimeSeriesProgressBar->setRange(0, abs(mEndTimeInput->value() - mStartTimeInput->value()) + 1);

  mTotalFrames = abs(mEndTimeInput->value() - mStartTimeInput->value()) + 1;

  mCaptureSettings->endTime = t;
}

void
RenderDialog::onSelectSaveDirectoryClicked()
{
  QString dir = QFileDialog::getExistingDirectory(this,
                                                  tr("Select Directory"),
                                                  mSaveDirectoryLabel->text(),
                                                  QFileDialog::ShowDirsOnly | QFileDialog::DontResolveSymlinks);
  if (!dir.isEmpty()) {
    mSaveDirectoryLabel->setText(dir);
    mCaptureSettings->outputDir = dir.toStdString();
  }
}

void
RenderDialog::onSaveFilePrefixChanged(const QString& value)
{
  mCaptureSettings->filenamePrefix = value.toStdString();
}
