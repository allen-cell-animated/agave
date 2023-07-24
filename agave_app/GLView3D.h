#pragma once

#include <memory>

#include "glad/glad.h"

#include "CameraController.h"
#include "glm.h"
#include "renderlib/CCamera.h"
#include "renderlib/gesture/gesture.h"

#include <QOpenGLWidget>
#include <QTimer>

class CStatus;
class ImageXYZC;
class QCamera;
class IRenderWindow;
class QRenderSettings;
class Scene;
namespace Serialize {
struct ViewerState;
}

/**
 * 3D GL view of an image with axes and gridlines.
 */
class GLView3D : public QOpenGLWidget
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
  GLView3D(QCamera* cam, QRenderSettings* qrs, RenderSettings* rs, QWidget* parent = 0);

  /// Destructor.
  ~GLView3D();

  QSize minimumSizeHint() const;
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

signals:
  void ChangedRenderer();

public slots:

  void OnUpdateCamera();
  void OnUpdateQRenderSettings(void);
  void OnUpdateRenderer(int);

public:
  std::shared_ptr<CStatus> getStatus();

  /// Resize the view.
  void resizeGL(int w, int h);

protected:
  /// Set up GL context and subsidiary objects.
  void initializeGL();

  /// Render the scene with the current view settings.
  void paintGL();

  void keyPressEvent(QKeyEvent* event);
  void mousePressEvent(QMouseEvent* event);
  void mouseReleaseEvent(QMouseEvent* event);
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
  void FitToScene();

  CCamera m_CCamera;
  CameraController m_cameraController;
  std::vector<CameraAnimation> m_cameraAnim;
  QCamera* m_qcamera;
  QRenderSettings* m_qrendersettings;

  double m_lastTimeCheck;
  double m_frameRate;
  int m_increments;

  /// Rendering timer.
  QTimer* m_etimer;

  /// Last mouse position.
  QPoint m_lastPos;

  RenderSettings* m_renderSettings;

  std::unique_ptr<IRenderWindow> m_renderer;
  int m_rendererType;

  Gesture m_gesture;
  Clock m_clock;
};
