#include "renderDialog.h"
#include "renderer.h"

#include "renderlib/Logging.h"
#include "renderlib/RenderGLPT.h"
#include "renderlib/command.h"

#include <QApplication>
#include <QButtonGroup>
#include <QCheckBox>
#include <QComboBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QImageWriter>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QMouseEvent>
#include <QPainter>
#include <QProgressBar>
#include <QPushButton>
#include <QSpinBox>
#include <QStackedWidget>
#include <QStandardPaths>
#include <QTimeEdit>
#include <QToolBar>
#include <QVBoxLayout>
#include <QWidget>

static const float ZOOM_STEP = 1.1f;

ImageDisplay::ImageDisplay(QWidget* parent)
  : QWidget(parent)
  , m_scale(1.0)
{
  m_pixmap = nullptr;
  setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
  m_pixmap = new QPixmap(256, 256);
  m_pixmap->fill(Qt::black);
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
    // can this processing step be skipped at all?

    // remove alphas from image for display here
    QImage im = image->convertToFormat(QImage::Format_RGB32);

    m_pixmap->convertFromImage(im, Qt::ColorOnly);
    m_rect = m_pixmap->rect();
    m_rect.translate(-m_rect.center());
  }
  update();
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

static QLabel*
makeGroupLabel(const std::string& text)
{
  QLabel* label = new QLabel(QString::fromStdString(text));
  label->setStyleSheet("font-weight: bold; font-size: 12px;");
  return label;
}

RenderDialog::RenderDialog(IRenderWindow* borrowedRenderer,
                           const RenderSettings& renderSettings,
                           const Scene& scene,
                           CCamera camera,
                           QOpenGLContext* glContext,
                           const LoadSpec& loadSpec,
                           CaptureSettings* captureSettings,
                           QWidget* parent)
  : m_renderer(borrowedRenderer)
  , m_renderSettings(renderSettings)
  , m_scene(scene)
  , m_camera(camera)
  , m_glContext(glContext)
  , m_loadSpec(loadSpec)
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
  setStyleSheet(R"(
QGroupBox
{
    font-size: 12px;
    /*font-weight: bold;*/
}
)");
  mImageView = new ImageDisplay(this);
  mRenderButton = new QPushButton("&Render", this);
  // mPauseRenderButton = new QPushButton("&Pause", this);
  mStopRenderButton = new QPushButton("&Stop", this);
  // mSaveButton = new QPushButton("&Save", this);

  mFrameProgressBar = new QProgressBar(this);
  if (mCaptureSettings->durationType == eRenderDurationType::SAMPLES) {
    mFrameProgressBar->setRange(0, mCaptureSettings->samples);
  } else {
    mFrameProgressBar->setRange(0, mCaptureSettings->duration);
  }

  mRenderDurationEdit = new QButtonGroup(this);
  QPushButton* samplesButton = new QPushButton(tr("Samples"), this);
  samplesButton->setCheckable(true);
  QPushButton* timeButton = new QPushButton(tr("Time"), this);
  timeButton->setCheckable(true);
  mRenderDurationEdit->addButton(samplesButton, eRenderDurationType::SAMPLES);
  mRenderDurationEdit->addButton(timeButton, eRenderDurationType::TIME);
  auto mapDurationTypeToUIIndex = std::map<eRenderDurationType, int>{
    { eRenderDurationType::SAMPLES, 0 },
    { eRenderDurationType::TIME, 1 },
  };
  mRenderDurationEdit->button(mCaptureSettings->durationType)->setChecked(true);
  // mRenderDurationEdit->setCurrentIndex(mapDurationTypeToUIIndex[mCaptureSettings->durationType]);

  mRenderSamplesEdit = new QSpinBox(this);
  mRenderSamplesEdit->setMinimum(1);
  // arbitrarily chosen
  mRenderSamplesEdit->setMaximum(65536);
  mRenderSamplesEdit->setValue(mCaptureSettings->samples);
  mRenderTimeEdit = new QTimeEdit(this);
  mRenderTimeEdit->setDisplayFormat("hh:mm:ss");
  int h = mCaptureSettings->duration / (60 * 60);
  int m = (mCaptureSettings->duration - h * 60 * 60) / 60;
  int s = (mCaptureSettings->duration - h * 60 * 60 - m * 60);
  mRenderTimeEdit->setTime(QTime(h, m, s));

  mWidth = mCaptureSettings->width;
  mHeight = mCaptureSettings->height;
  mAspectRatio = (float)mWidth / (float)mHeight;

  m_camera.m_Film.m_Resolution.SetResX(mWidth);
  m_camera.m_Film.m_Resolution.SetResY(mHeight);

  mWidthInput = new QSpinBox(this);
  mWidthInput->setMaximum(4096);
  mWidthInput->setMinimum(2);
  mWidthInput->setValue(mWidth);
  mHeightInput = new QSpinBox(this);
  mHeightInput->setMaximum(4096);
  mHeightInput->setMinimum(2);
  mHeightInput->setValue(mHeight);
  mLockAspectRatio = new QPushButton(QIcon(":/icons/lock.png"), "", this);
  mLockAspectRatio->setCheckable(true);
  mLockAspectRatio->setChecked(true);
  mResolutionPresets = new QComboBox(this);
  mResolutionPresets->addItem("Choose Preset...");
  mResolutionPresets->addItem("Main window");
  for (int i = 0; i < sizeof(resolutionPresets) / sizeof(ResolutionPreset); i++) {
    mResolutionPresets->addItem(resolutionPresets[i].label);
  }

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

  mSelectSaveDirectoryButton = new QPushButton("...", this);

  mAutosaveCheckbox = new QCheckBox("Autosave", this);
  mAutosaveCheckbox->setChecked(true);
  mAutosaveCheckbox->setVisible(false);
  mAutosaveCheckbox->setEnabled(false);

  mSaveDirectoryLabel = new QLabel(QString::fromStdString(mCaptureSettings->outputDir), this);
  mSaveFilePrefix = new QLineEdit(QString::fromStdString(mCaptureSettings->filenamePrefix), this);

  mToolbar = new QToolBar(mImageView);
  mToolbar->addAction("+", this, &RenderDialog::onZoomInClicked);
  mToolbar->addAction("-", this, &RenderDialog::onZoomOutClicked);
  mToolbar->addAction("[ ]", this, &RenderDialog::onZoomFitClicked);

  connect(mRenderButton, &QPushButton::clicked, this, &RenderDialog::render);
  // connect(mPauseRenderButton, &QPushButton::clicked, this, &RenderDialog::pauseRendering);
  connect(mStopRenderButton, &QPushButton::clicked, this, &RenderDialog::onStopButtonClick);
  // connect(mSaveButton, &QPushButton::clicked, this, &RenderDialog::save);
  connect(mResolutionPresets, SIGNAL(currentIndexChanged(int)), this, SLOT(onResolutionPreset(int)));
  connect(mWidthInput, QOverload<int>::of(&QSpinBox::valueChanged), this, &RenderDialog::updateWidth);
  connect(mHeightInput, QOverload<int>::of(&QSpinBox::valueChanged), this, &RenderDialog::updateHeight);
  connect(mRenderSamplesEdit, SIGNAL(valueChanged(int)), this, SLOT(updateRenderSamples(int)));
  connect(mRenderTimeEdit, SIGNAL(timeChanged(const QTime&)), this, SLOT(updateRenderTime(const QTime&)));
  connect(mRenderDurationEdit, SIGNAL(idClicked(int)), this, SLOT(onRenderDurationTypeChanged(int)));
  //  connect(mRenderDurationEdit, SIGNAL(currentIndexChanged(int)), this, SLOT(onRenderDurationTypeChanged(int)));
  connect(mStartTimeInput, QOverload<int>::of(&QSpinBox::valueChanged), this, &RenderDialog::onStartTimeChanged);
  connect(mEndTimeInput, QOverload<int>::of(&QSpinBox::valueChanged), this, &RenderDialog::onEndTimeChanged);
  connect(mSelectSaveDirectoryButton, &QPushButton::clicked, this, &RenderDialog::onSelectSaveDirectoryClicked);
  connect(mSaveFilePrefix, &QLineEdit::textChanged, this, &RenderDialog::onSaveFilePrefixChanged);

  QVBoxLayout* outputResolutionLayout = new QVBoxLayout();
  QHBoxLayout* topButtonsLayout = new QHBoxLayout();
  topButtonsLayout->addWidget(new QLabel(tr("X:")), 0);
  topButtonsLayout->addWidget(mWidthInput, 1);
  topButtonsLayout->addWidget(new QLabel(tr("Y:")), 0);
  topButtonsLayout->addWidget(mHeightInput, 1);
  topButtonsLayout->addWidget(mLockAspectRatio, 0);
  outputResolutionLayout->addWidget(makeGroupLabel("Output Resolution"));
  outputResolutionLayout->addWidget(mResolutionPresets);
  outputResolutionLayout->addLayout(topButtonsLayout);

  QHBoxLayout* timeHLayout = new QHBoxLayout();
  timeHLayout->addWidget(new QLabel(tr("Start:")), 0);
  timeHLayout->addWidget(mStartTimeInput, 1);
  timeHLayout->addWidget(new QLabel(tr("End:")), 0);
  timeHLayout->addWidget(mEndTimeInput, 1);
  QVBoxLayout* timeLayout = new QVBoxLayout();
  timeLayout->addWidget(makeGroupLabel("Time Series"));
  timeLayout->addLayout(timeHLayout);

  //  QHBoxLayout* saveFileLayout = new QHBoxLayout();
  // saveFileLayout->addWidget(mSaveFilePrefix, 1);
  QHBoxLayout* saveDirLayout = new QHBoxLayout();
  saveDirLayout->addWidget(mSaveDirectoryLabel, 2);
  saveDirLayout->addWidget(mSelectSaveDirectoryButton, 1);
  QFormLayout* saveSettingsLayout = new QFormLayout();
  saveSettingsLayout->addRow(makeGroupLabel("Output File"));
  saveSettingsLayout->addRow(tr("File Name:"), mSaveFilePrefix);
  saveSettingsLayout->addRow(tr("Location:"), saveDirLayout);

  QHBoxLayout* durationsHLayout = new QHBoxLayout();
  durationsHLayout->addWidget(mRenderDurationEdit->button(eRenderDurationType::SAMPLES), 0);
  durationsHLayout->addWidget(mRenderDurationEdit->button(eRenderDurationType::TIME), 0);

  mRenderDurationSettings = new QStackedWidget(this);

  QFormLayout* durationsHLayoutTime = new QFormLayout();
  durationsHLayoutTime->addRow(tr("Time:"), mRenderTimeEdit);
  QWidget* durationSettingsTime = new QWidget();
  durationSettingsTime->setLayout(durationsHLayoutTime);

  QFormLayout* durationsHLayoutSamples = new QFormLayout();
  durationsHLayoutSamples->addRow(tr("Samples:"), mRenderSamplesEdit);
  QWidget* durationSettingsSamples = new QWidget();
  durationSettingsSamples->setLayout(durationsHLayoutSamples);

  mRenderDurationSettings->addWidget(durationSettingsTime);
  mRenderDurationSettings->addWidget(durationSettingsSamples);
  // initialize
  setRenderDurationType(mCaptureSettings->durationType);

  QVBoxLayout* durationsLayout = new QVBoxLayout();
  durationsLayout->addWidget(makeGroupLabel("Image Quality"));
  durationsLayout->addLayout(durationsHLayout);
  durationsLayout->addWidget(mRenderDurationSettings);

  QHBoxLayout* bottomButtonsLayout = new QHBoxLayout();
  bottomButtonsLayout->addWidget(mRenderButton);
  // bottomButtonslayout->addWidget(mPauseRenderButton);
  bottomButtonsLayout->addWidget(mStopRenderButton);
  mStopRenderButton->setVisible(false);
  // bottomButtonslayout->addWidget(mSaveButton);

  static const int MAX_CONTROLS_WIDTH = 400;

  QGroupBox* groupBox0 = new QGroupBox();
  groupBox0->setMaximumWidth(MAX_CONTROLS_WIDTH);
  groupBox0->setLayout(outputResolutionLayout);

  QGroupBox* groupBox1 = new QGroupBox();
  groupBox1->setMaximumWidth(MAX_CONTROLS_WIDTH);
  groupBox1->setLayout(timeLayout);
  groupBox1->setEnabled(scene.m_timeLine.maxTime() > 0);

  QGroupBox* groupBox2 = new QGroupBox();
  groupBox2->setMaximumWidth(MAX_CONTROLS_WIDTH);
  groupBox2->setLayout(durationsLayout);

  QGroupBox* groupBox3 = new QGroupBox();
  groupBox3->setMaximumWidth(MAX_CONTROLS_WIDTH);
  groupBox3->setLayout(saveSettingsLayout);

  QGroupBox* groupBox4 = new QGroupBox();
  groupBox4->setMaximumWidth(MAX_CONTROLS_WIDTH);
  groupBox4->setLayout(bottomButtonsLayout);

  QVBoxLayout* controlsLayout = new QVBoxLayout();
  controlsLayout->addWidget(groupBox0);
  controlsLayout->addWidget(groupBox1);
  controlsLayout->addWidget(groupBox2);
  controlsLayout->addWidget(groupBox3);
  controlsLayout->addWidget(groupBox4);
  controlsLayout->setSpacing(0);
  controlsLayout->setContentsMargins(0, 0, 0, 0);

  QGroupBox* controlsGroupBox = new QGroupBox();
  controlsGroupBox->setLayout(controlsLayout);

  QVBoxLayout* viewLayout = new QVBoxLayout();
  viewLayout->addWidget(mImageView);
  viewLayout->addWidget(mToolbar);

  QHBoxLayout* mainDialogLayout = new QHBoxLayout();
  mainDialogLayout->addWidget(controlsGroupBox, 1);
  mainDialogLayout->addLayout(viewLayout, 3);

  QGroupBox* progressGroup = new QGroupBox();
  QFormLayout* progressLayout = new QFormLayout();
  progressLayout->addRow(makeGroupLabel("Rendering " + loadSpec.getFilename()));
  progressLayout->addRow(tr("Frame Progress"), mFrameProgressBar);
  progressLayout->addRow(tr("Total Progress"), mTimeSeriesProgressBar);
  // progressLayout->setContentsMargins(0, 0, 0, 0);
  progressLayout->setFieldGrowthPolicy(QFormLayout::ExpandingFieldsGrow);
  progressGroup->setLayout(progressLayout);

  QVBoxLayout* reallyMainDialogLayout = new QVBoxLayout();
  reallyMainDialogLayout->addLayout(mainDialogLayout);
  reallyMainDialogLayout->addWidget(progressGroup);

  setLayout(reallyMainDialogLayout);
}

void
RenderDialog::save()
{
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
  mRenderButton->setVisible(false);
  mStopRenderButton->setVisible(true);
  LOG_INFO << "Render button clicked";
  if (!this->m_renderThread || m_renderThread->isFinished()) {

    resetProgress();

    if (!m_renderThread) {

      m_renderThread = new Renderer("Render dialog render thread ", this, m_mutex);

      // queued across thread boundary.  requestProcessed is called from another thread, asynchronously.
      connect(m_renderThread, &Renderer::requestProcessed, this, &RenderDialog::onRenderRequestProcessed);
      connect(m_renderThread, &Renderer::finished, this, &RenderDialog::onRenderThreadFinished);
    }

    // now get our rendering resources into this Renderer object
    m_renderThread->configure(m_renderer, m_renderSettings, m_scene, m_camera, m_loadSpec, m_glContext);

    onZoomFitClicked();
    // first time in, set up stream mode and give the first draw request
    resumeRendering();

    LOG_INFO << "Starting render thread...";
    m_glContext->moveToThread(m_renderThread);
    m_renderThread->start();
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

  if (req == nullptr) {
    LOG_DEBUG << "render callback after no request processed";
    return;
  }
  if (!m_renderThread || m_renderThread->isFinished() || m_renderThread->isInterruptionRequested()) {
    LOG_DEBUG << "received sample after render terminated";
  }
  if (mTimeSeriesProgressBar->value() >= mTimeSeriesProgressBar->maximum()) {
    LOG_DEBUG << "received frame after timeline completed";
    return;
  }

  // increment our time counter
  this->m_frameRenderTime += req->getActualDuration();

  // increment progress
  if (mRenderDurationType == eRenderDurationType::SAMPLES) {
    mFrameProgressBar->setValue(mFrameProgressBar->value() + 1);
  } else {
    // nano to seconds.  render durations in the dialog are specified in seconds.
    mFrameProgressBar->setValue(m_frameRenderTime / (1000 * 1000 * 1000));
  }

  // did a frame finish?
  if (mFrameProgressBar->value() >= mFrameProgressBar->maximum()) {
    // we just received the last sample.
    // however, another sample was already enqueued!!!!
    // so we know we will have one sample to discard.

    LOG_DEBUG << mFrameProgressBar->value() << " images received";
    LOG_DEBUG << "Progress " << mFrameProgressBar->value() << " / " << mFrameProgressBar->maximum();
    LOG_DEBUG << "frame " << mFrameNumber << " progress completed";

    // update display with finished frame
    this->setImage(&image);

    // save image
    if (mAutosaveCheckbox->isChecked()) {
      QString autosavePath = mSaveDirectoryLabel->text();
      QDir d(autosavePath);
      // TODO set up autosave path when we start rendering
      bool pathOk = d.mkpath(autosavePath);
      if (!pathOk) {
        LOG_ERROR << "Failed to make path " << autosavePath.toStdString();
      }
      // save!
      QString filename = mSaveFilePrefix->text() + QString("_%1.png").arg(mFrameNumber, 4, 10, QChar('0'));
      QFileInfo fileInfo(d, filename);
      QString saveFilePath = fileInfo.absoluteFilePath();

      // TODO don't throw away alpha - rethink how image is composited with background color
      QImage im = image.convertToFormat(QImage::Format_RGB32);
      bool ok = im.save(saveFilePath);
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
      LOG_DEBUG << "all frames completed.  ending render";
      stopRendering();
    } else {
      LOG_DEBUG << "reset frame progress for next frame";
      // reset frame progress and render time
      mFrameProgressBar->setValue(0);
      m_frameRenderTime = 0;

      // set up for next frame!
      // this typecast should be eliminated,
      // possibly a GetRenderSettings abstract command or something
      RenderGLPT* r = dynamic_cast<RenderGLPT*>(m_renderer);
      r->getRenderSettings().SetNoIterations(0);

      LOG_DEBUG << "queueing setTime " << mFrameNumber << " command ";
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
  // note this leaves the render thread alive and well
  if (m_renderThread) {
    m_renderThread->setStreamMode(0);
  }
}

void
RenderDialog::stopRendering()
{
  if (m_renderThread) {
    endRenderThread();
  }
}

void
RenderDialog::endRenderThread()
{
  pauseRendering();
  if (m_renderThread && m_renderThread->isRunning()) {
    m_renderThread->requestInterruption();
    m_renderThread->wakeUp();
    // we need to ensure that the render thread is not trying to make calls back into this thread
    m_renderThread->wait();
  }
}

void
RenderDialog::resumeRendering()
{
  if (m_renderThread) {
    m_renderThread->setStreamMode(1);

    std::vector<Command*> cmd;

    // 1. pick up any resolution changes?
    SetResolutionCommandD resdata;
    resdata.m_x = mWidth;
    resdata.m_y = mHeight;
    cmd.push_back(new SetResolutionCommand(resdata));

    // 2. make sure we set the right time
    SetTimeCommandD timedata;
    timedata.m_time = mFrameNumber;
    cmd.push_back(new SetTimeCommand(timedata));

    // 3. the redraw request
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
    m_renderThread = nullptr;
  }
  QDialog::done(r);
}

void
RenderDialog::onResolutionPreset(int index)
{
  if (index > 1) {
    // find preset res and set w/h
    const ResolutionPreset& preset = resolutionPresets[index - 2];
    mAspectRatio = (float)preset.w / (float)preset.h;
    mWidthInput->setValue((preset.w));
    mHeightInput->setValue((preset.h));
  } else if (index == 1) {
    // get xy from the main window size.
  }
  // restore index 0
  mResolutionPresets->setCurrentIndex(0);
}

void
RenderDialog::updateWidth(int w)
{
  mWidth = w;
  m_camera.m_Film.m_Resolution.SetResX(w);
  mCaptureSettings->width = w;

  if (mLockAspectRatio->isChecked()) {
    mHeight = (int)(mWidth / mAspectRatio);
    mHeightInput->blockSignals(true);
    mHeightInput->setValue(mHeight);
    mHeightInput->blockSignals(false);
    m_camera.m_Film.m_Resolution.SetResY(mHeight);
    mCaptureSettings->height = mHeight;
  } else {
    mAspectRatio = (float)mWidth / (float)mHeight;
  }

  resetProgress();
}

void
RenderDialog::updateHeight(int h)
{
  mHeight = h;
  m_camera.m_Film.m_Resolution.SetResY(h);
  mCaptureSettings->height = h;

  if (mLockAspectRatio->isChecked()) {
    mWidth = (int)(mHeight * mAspectRatio);
    mWidthInput->blockSignals(true);
    mWidthInput->setValue(mWidth);
    mWidthInput->blockSignals(false);
    m_camera.m_Film.m_Resolution.SetResX(mWidth);
    mCaptureSettings->width = mWidth;
  } else {
    mAspectRatio = (float)mWidth / (float)mHeight;
  }

  resetProgress();
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
  mRenderDurationSettings->setCurrentIndex(type);
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
  mFrameProgressBar->setValue(0);

  m_frameRenderTime = 0; // FIX per frame render time vs total render time elapsed

  mTimeSeriesProgressBar->reset();
  mTimeSeriesProgressBar->setValue(0);
  mFrameNumber = mStartTimeInput->value();
}

void
RenderDialog::onRenderDurationTypeChanged(int index)
{
  // get userdata from value
  eRenderDurationType type = (eRenderDurationType)index;
  setRenderDurationType(type);
  if (type == eRenderDurationType::SAMPLES) {
    updateRenderSamples(mRenderSamplesEdit->value());
  } else {
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

void
RenderDialog::onRenderThreadFinished()
{
  m_renderThread->deleteLater();
  m_renderThread = nullptr;
}

bool
RenderDialog::getUserCancelConfirmation()
{
  QMessageBox::StandardButton btn =
    QMessageBox::question(this, "Cancel Render?", "Are you sure you want to cancel the render currently in progress?");
  if (btn == QMessageBox::Yes) {
    return true;
  } else {
    return false;
  }
}

bool
RenderDialog::isRenderInProgress()
{
  if (!m_renderThread || m_renderThread->isFinished() || m_renderThread->isInterruptionRequested()) {
    return false;
  }
  if (mTimeSeriesProgressBar->value() >= mTimeSeriesProgressBar->maximum()) {
    return false;
  }
  if (mFrameProgressBar->value() >= mFrameProgressBar->maximum()) {
    return false;
  }
  return true;
}

void
RenderDialog::closeEvent(QCloseEvent* event)
{
  LOG_DEBUG << "closeEvent()";

  if (isRenderInProgress()) {
    if (getUserCancelConfirmation()) {
      event->accept();
      QDialog::closeEvent(event);
    } else {
      event->ignore();
      // do not close the dialog!
    }
  } else {
    event->accept();
    QDialog::closeEvent(event);
  }
}

void
RenderDialog::onStopButtonClick()
{
  // note this does not pause/resume while waiting for confirmation
  if (isRenderInProgress()) {
    if (getUserCancelConfirmation()) {
      stopRendering();
      mRenderButton->setVisible(true);
      mStopRenderButton->setVisible(false);
    }
  } else {
    stopRendering();
    mRenderButton->setVisible(true);
    mStopRenderButton->setVisible(false);
  }
}