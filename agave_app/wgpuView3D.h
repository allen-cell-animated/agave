#pragma once

#include <memory>

#include "glm.h"

#include "renderlib/ViewerWindow.h"
#include "renderlib/gesture/gesture.h"
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

// does the actual wgpu stuff
// for some reason we need this to be a child of the main view
class wgpuCanvas : public QWidget
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
  wgpuCanvas(QCamera* cam, QRenderSettings* qrs, RenderSettings* rs, QWidget* parent = 0);

  /// Destructor.
  ~wgpuCanvas();

  // dummy to make agaveGui happy
  void doneCurrent() {}
  QOpenGLContext* context() { return nullptr; }
  void resizeGL(int w, int h);

signals:
  void ChangedRenderer();

public:
  std::shared_ptr<CStatus> getStatus();

protected:
  /// Set up GL context and subsidiary objects.
  void initializeGL(WGPUTextureView nextTexture);

  /// Render the scene with the current view settings.
  void paintGL(WGPUTextureView nextTexture);

  void resizeEvent(QResizeEvent* event);
  void paintEvent(QPaintEvent* event);

  virtual QPaintEngine* paintEngine() const override { return nullptr; }

private:
  /// Rendering timer.
  QElapsedTimer m_etimer;

  /// Last mouse position.
  QPoint m_lastPos;

  bool m_initialized;
  bool m_fakeHidden;
  void render();
  void invokeUserPaint(WGPUTextureView nextTexture);
  WGPUTextureFormat m_swapChainFormat;
  WGPUSurface m_surface;
  WGPUDevice m_device;

  WGPURenderPipeline m_pipeline;
};

class WgpuView3D : public QWidget
{
  Q_OBJECT;

public:
  WgpuView3D(QCamera* cam, QRenderSettings* qrs, RenderSettings* rs, QWidget* parent = 0);
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

  const CCamera& getCamera() { return m_viewerWindow->m_CCamera; }

  void fromViewerState(const Serialize::ViewerState& s);

  QPixmap capture();
  QImage captureQimage();

  // DANGER this must NOT outlive the GLView3D
  IRenderWindow* borrowRenderer() { return m_viewerWindow->m_renderer.get(); }

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
  wgpuCanvas* m_canvas;

  QCamera* m_qcamera;
  QRenderSettings* m_qrendersettings;

  ViewerWindow* m_viewerWindow;
};
