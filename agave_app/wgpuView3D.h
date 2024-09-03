#pragma once

#include <memory>

#include "glm.h"

#include "renderlib/ViewerWindow.h"
#include "renderlib/gesture/gesture.h"
#include "renderlib_wgpu/renderlib_wgpu.h"

#include <QElapsedTimer>
#include <QHBoxLayout>
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
  QSize minimumSizeHint() const override;

  /**
   * Get window size hint.
   *
   * @returns the size hint.
   */
  QSize sizeHint() const override;

  void initCameraFromImage(Scene* scene);
  void retargetCameraForNewVolume(Scene* scene);

  void toggleCameraProjection();
  void toggleAreaLightRotateControls();

  void onNewImage(Scene* scene);

  const CCamera& getCamera() { return m_viewerWindow->m_CCamera; }

  void fromViewerState(const Serialize::ViewerState& s);

  QPixmap capture();
  QImage captureQimage();

  // DANGER this must NOT outlive the GLView3D
  ViewerWindow* borrowRenderer() { return m_viewerWindow; }

  void pauseRenderLoop();
  void restartRenderLoop();

  // dummy to make agaveGui happy
  void doneCurrent() {}
  QOpenGLContext* context() { return nullptr; }
  void resizeGL(int w, int h);
  void FitToScene(float transitionDurationSeconds = 0.0f);

signals:
  void ChangedRenderer();

public slots:

  void OnUpdateCamera();
  void OnUpdateQRenderSettings(void);
  void OnUpdateRenderer(int);

public:
  std::shared_ptr<CStatus> getStatus();

protected:
  void mousePressEvent(QMouseEvent* event) override;
  void mouseReleaseEvent(QMouseEvent* event) override;
  void mouseMoveEvent(QMouseEvent* event) override;
  void wheelEvent(QWheelEvent* event) override;
  void keyPressEvent(QKeyEvent* event) override;

private:
  QCamera* m_qcamera;
  QRenderSettings* m_qrendersettings;

  ViewerWindow* m_viewerWindow;

  /// Rendering timer.
  QTimer* m_etimer;

  /// Last mouse position.
  QPoint m_lastPos;

  bool m_initialized;
  bool m_fakeHidden;
  void render();
  void renderWindowContents(WGPUTextureView nextTexture);
  WGPUDevice m_device;
  WGPUQueue m_queue;

  WgpuWindowContext m_windowContext;

  WGPURenderPipeline m_pipeline;

  QWidget* m_canvas;

  enum AREALIGHT_MODE
  {
    NONE,
    ROT,
    TRANS
  } m_areaLightMode = AREALIGHT_MODE::NONE;

protected:
  /// Set up GL context and subsidiary objects.
  void initializeGL();

  void resizeEvent(QResizeEvent* event) override;
  void paintEvent(QPaintEvent* event) override;

  virtual QPaintEngine* paintEngine() const override { return nullptr; }
};

class WgpuCanvas : public QWidget
{
  Q_OBJECT;

public:
  WgpuCanvas(QCamera* cam, QRenderSettings* qrs, RenderSettings* rs, QWidget* parent = 0);
  ~WgpuCanvas() { delete m_view; }

  // make sure every time this updates, the child updates
  // void update() override { m_view->update(); }

  std::shared_ptr<CStatus> getStatus() { return m_view->getStatus(); }
  QImage captureQimage() { return m_view->captureQimage(); }
  void pauseRenderLoop() { m_view->pauseRenderLoop(); }
  void restartRenderLoop() { m_view->restartRenderLoop(); }
  void doneCurrent() {}
  // DANGER this must NOT outlive the WgpuCanvas
  ViewerWindow* borrowRenderer() { return m_view->borrowRenderer(); }
  const CCamera& getCamera() { return m_view->getCamera(); }
  QOpenGLContext* context() { return nullptr; }
  void resizeGL(int w, int h) { m_view->resizeGL(w, h); }
  void FitToScene(float transitionDurationSeconds = 0.0f) { m_view->FitToScene(transitionDurationSeconds); }
  void initCameraFromImage(Scene* scene) { m_view->initCameraFromImage(scene); }
  void retargetCameraForNewVolume(Scene* scene) { m_view->retargetCameraForNewVolume(scene); }

  void onNewImage(Scene* scene) { m_view->onNewImage(scene); }
  void toggleCameraProjection() { m_view->toggleCameraProjection(); }
  void toggleAreaLightRotateControls() { m_view->toggleAreaLightRotateControls(); }
  void fromViewerState(const Serialize::ViewerState& s) { m_view->fromViewerState(s); }

signals:
  void ChangedRenderer();

public slots:
  void OnChangedRenderer() { emit ChangedRenderer(); }

private:
  WgpuView3D* m_view;
  QHBoxLayout* m_layout;
};
