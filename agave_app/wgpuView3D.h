#pragma once

#include <memory>

#include "CameraController.h"
#include "glm.h"
#include "renderlib/CCamera.h"

#include "renderlib_wgpu/renderlib_wgpu.h"

#include <QElapsedTimer>
#include <QWidget>

class QOpenGLContext;

class CStatus;
class ImageXYZC;
class QCamera;
class IRenderWindow;
class QRenderSettings;
class Scene;
namespace Serialize {
struct ViewerState;
};

/**
 * 3D GL view of an image with axes and gridlines.
 */
class WgpuView3D : public QWidget
{
  Q_OBJECT

public:
  /**
   * Create a 3D image view.
   *
   * The size and position will be taken from the specified image.
   *
   * @param reader the image reader.
   * @param series the image series.
   * @param parent the parent of this object.
   */
  WgpuView3D(QCamera* cam, QRenderSettings* qrs, RenderSettings* rs, QWidget* parent = 0);

  /// Destructor.
  ~WgpuView3D();

  /**
   * Get window minimum size hint.
   *
   * @returns the size hint.
   */
  QSize minimumSizeHint() const;

  /**
   * Get window size hint.
   *
   * @returns the size hint.
   */
  QSize sizeHint() const;

  void initCameraFromImage(Scene* scene);
  void toggleCameraProjection();

  void onNewImage(Scene* scene);

  const CCamera& getCamera() { return m_CCamera; }

  void fromViewerState(const Serialize::ViewerState& s);

  QPixmap capture();
  QImage captureQimage();

  // DANGER this must NOT outlive the GLView3D
  IRenderWindow* borrowRenderer() { return m_renderer.get(); }

  void pauseRenderLoop();
  void restartRenderLoop();

  // dummy to make agaveGui happy
  void doneCurrent() {}
  QOpenGLContext* context() { return nullptr; }
  void resizeGL(int w, int h);

signals:
  void ChangedRenderer();

public slots:

  void OnUpdateCamera();
  void OnUpdateQRenderSettings(void);
  void OnUpdateRenderer(int);

public:
  std::shared_ptr<CStatus> getStatus();

protected:
  /// Set up GL context and subsidiary objects.
  void initializeGL();

  /// Render the scene with the current view settings.
  void paintGL(WGPUTextureView nextTexture);

  void resizeEvent(QResizeEvent* event);
  void paintEvent(QPaintEvent* event);

  virtual QPaintEngine* paintEngine() const override { return nullptr; }

  /**
   * Handle mouse button press events.
   *
   * Action depends upon the mouse behaviour mode.
   *
   * @param event the event to handle.
   */
  void mousePressEvent(QMouseEvent* event);
  void mouseReleaseEvent(QMouseEvent* event);

  /**
   * Handle mouse button movement events.
   *
   * Action depends upon the mouse behaviour mode.
   *
   * @param event the event to handle.
   */
  void mouseMoveEvent(QMouseEvent* event);

  /**
   * Handle timer events.
   *
   * Used to update scene properties and trigger a render pass.
   *
   * @param event the event to handle.
   */
  void timerEvent(QTimerEvent* event);

private:
  CCamera m_CCamera;
  CameraController m_cameraController;
  QCamera* m_qcamera;
  QRenderSettings* m_qrendersettings;

  /// Rendering timer.
  QElapsedTimer m_etimer;

  /// Last mouse position.
  QPoint m_lastPos;

  RenderSettings* m_renderSettings;

  std::unique_ptr<IRenderWindow> m_renderer;
  int m_rendererType;

  bool m_initialized;
  bool m_fakeHidden;
  void render();
  void invokeUserPaint(WGPUTextureView nextTexture);
  WGPUSwapChain m_swapChain;
  WGPUTextureFormat m_swapChainFormat;
  WGPUSurface m_surface;
  WGPUDevice m_device;

  WGPURenderPipeline m_pipeline;
  QWidget* m_canvas;
};

class wgpuCanvas : public QWidget
{
  Q_OBJECT;

public:
  wgpuCanvas(QWidget* parent = nullptr);
  ~wgpuCanvas();

private:
  WgpuView3D* m_view;
};
