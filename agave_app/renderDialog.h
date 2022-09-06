#pragma once

#include "renderlib/CCamera.h"

#include <QDialog>
#include <QMutex>

class QComboBox;
class QImage;
class QOpenGLContext;
class QWidget;
class QLineEdit;
class QPixmap;
class QProgressBar;
class QPushButton;
class QSpinBox;
class QTimeEdit;

class IRenderWindow;
class Renderer;
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

private:
  QPixmap* m_pixmap;
  QImage* m_image;

protected:
  void paintEvent(QPaintEvent* event) override;
};

enum eRenderDurationType
{
  TIME = 0,
  SAMPLES = 1
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
  void updateRenderTime(QTime t);

signals:
  void setRenderResolution(int x, int y);

private:
  QMutex m_mutex;
  QOpenGLContext* m_glContext;
  Renderer* m_renderThread;
  IRenderWindow* m_renderer;
  const RenderSettings& m_renderSettings;
  const Scene& m_scene;
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
  QProgressBar* mProgressBar;
  QLineEdit* mRenderDurationEdit;
  QSpinBox* mRenderSamplesEdit;
  QTimeEdit* mRenderTimeEdit;

  int mWidth;
  int mHeight;
  qint64 m_totalRenderTime;
  // TODO controls to put in a render dialog:
  // save button
  // play controls for time series / anim sequence
  // cancel button to stop everything
  // termination criteria
  // - how many iterations
  // - how many seconds
  // - render until stopped
  // xy resolution

  // "quick render" means render image at current settings and exit
  void endRenderThread();
  void setRenderDurationType(eRenderDurationType type);
  eRenderDurationType mRenderDurationType;
  void resetProgress();
};
