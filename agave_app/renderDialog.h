#pragma once

#include "renderlib/CCamera.h"

#include <QDialog>

class QImage;
class QWidget;
class QPushButton;

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

private:
  QImage* m_image;

protected:
  void paintEvent(QPaintEvent* event) override;
};

class RenderDialog : public QDialog
{
  Q_OBJECT

public:
  RenderDialog(IRenderWindow* borrowedRenderer,
               const RenderSettings& renderSettings,
               const Scene& scene,
               CCamera camera,
               QWidget* parent = Q_NULLPTR);

  void setImage(QImage* image);
  void stop();
private slots:
  void render();

private:
  Renderer* m_renderThread;
  IRenderWindow* m_renderer;
  const RenderSettings& m_renderSettings;
  const Scene& m_scene;
  CCamera m_camera;

  ImageDisplay* mImageView; // ? or a GLView3D?
  QPushButton* mRenderButton;
  QPushButton* mCloseButton;

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
};
