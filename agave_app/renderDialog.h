#pragma once

#include "renderlib/CCamera.h"

#include <QDialog>
#include <QMutex>
#include <QStandardPaths>

class QComboBox;
class QImage;
class QOpenGLContext;
class QWidget;
class QLabel;
class QLineEdit;
class QPixmap;
class QProgressBar;
class QPushButton;
class QSpinBox;
class QTimeEdit;
class QToolBar;

class IRenderWindow;
class Renderer;
class RenderRequest;
class RenderSettings;
class Scene;

class ImageDisplay : public QWidget
{
  Q_OBJECT
public:
  ImageDisplay(QWidget* parent = 0);
  ~ImageDisplay();
  void setImage(QImage* image);
  void save(QString filename);

  void scale(qreal s);
  void setScale(qreal s);
  void fit(int w, int h);

private:
  QPixmap* m_pixmap;
  QImage* m_image;

  QRectF m_rect;
  QPointF m_reference;
  QPointF m_delta;
  qreal m_scale = 1.0;

  void mousePressEvent(QMouseEvent* event) override;
  void mouseMoveEvent(QMouseEvent* event) override;
  void mouseReleaseEvent(QMouseEvent*) override;

protected:
  void paintEvent(QPaintEvent* event) override;
};

// serialized so permanent?
enum eRenderDurationType
{
  TIME = 0,
  SAMPLES = 1
};

struct CaptureSettings
{
  std::string outputDir;
  std::string filenamePrefix;
  int width;
  int height;
  int samples;
  int duration; // in seconds
  eRenderDurationType durationType;
  int startTime;
  int endTime;

  CaptureSettings()
  {
    // defaults!
    QString docs = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
    outputDir = docs.toStdString();

    filenamePrefix = "frame";
    width = 640;
    height = 480;
    samples = 32;
    duration = 10;
    durationType = SAMPLES;
    startTime = 0;
    endTime = 0;
  }
};

class RenderDialog : public QDialog
{
  Q_OBJECT

public:
  RenderDialog(IRenderWindow* borrowedRenderer,
               const RenderSettings& renderSettings,
               const Scene& scene,
               CCamera camera,
               QOpenGLContext* glContext,
               std::string volumeFilePath,
               CaptureSettings* captureSettings,
               QWidget* parent = Q_NULLPTR);

  void setImage(QImage* image);
  void done(int r);
  int getXResolution();
  int getYResolution();

private slots:
  void render();
  void save();
  void pauseRendering();
  void stopRendering();
  void resumeRendering();
  void onResolutionPreset(int index);
  void updateWidth(int w);
  void updateHeight(int h);
  void updateRenderSamples(int s);
  void updateRenderTime(const QTime& t);
  void onRenderDurationTypeChanged(int index);

private:
  QMutex m_mutex;
  QOpenGLContext* m_glContext;
  Renderer* m_renderThread;
  IRenderWindow* m_renderer;
  const RenderSettings& m_renderSettings;
  const Scene& m_scene;
  std::string mVolumeFilePath;
  // reference that I don't own
  CaptureSettings* mCaptureSettings;
  CCamera m_camera;

  ImageDisplay* mImageView; // ? or a GLView3D?
  QPushButton* mRenderButton;
  QPushButton* mPauseRenderButton;
  QPushButton* mStopRenderButton;
  QPushButton* mSaveButton;
  QPushButton* mCloseButton;
  QSpinBox* mWidthInput;
  QSpinBox* mHeightInput;
  QComboBox* mResolutionPresets;
  QSpinBox* mStartTimeInput;
  QSpinBox* mEndTimeInput;

  QPushButton* mSelectSaveDirectoryButton;
  QLabel* mSaveDirectoryLabel;
  QLineEdit* mSaveFilePrefix;

  QProgressBar* mFrameProgressBar;
  QProgressBar* mTimeSeriesProgressBar;
  QComboBox* mRenderDurationEdit;
  QSpinBox* mRenderSamplesEdit;
  QTimeEdit* mRenderTimeEdit;
  QPushButton* mZoomInButton;
  QPushButton* mZoomOutButton;
  QPushButton* mZoomFitButton;

  QToolBar* mToolbar;

  int mWidth;
  int mHeight;
  qint64 m_frameRenderTime;
  // TODO controls to put in a render dialog:
  // save button
  // play controls for time series / anim sequence

  void endRenderThread();
  void setRenderDurationType(eRenderDurationType type);
  eRenderDurationType mRenderDurationType;
  void resetProgress();

  void onRenderRequestProcessed(RenderRequest* req, QImage image);
  void onZoomInClicked();
  void onZoomOutClicked();
  void onZoomFitClicked();

  int mFrameNumber;
  int mTotalFrames;

  void onStartTimeChanged(int);
  void onEndTimeChanged(int);

  void onSelectSaveDirectoryClicked();
  void onSaveFilePrefixChanged(const QString& value);
  void onRenderThreadFinished();
};
