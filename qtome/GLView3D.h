#pragma once

#include <memory>

#include "glad/glad.h"

#include "CameraController.h"
#include "glm.h"
#include "renderlib/CCamera.h"

#include <QElapsedTimer>
#include <QOpenGLWidget>

class CStatus;
class ImageXYZC;
class QCamera;
class IRenderWindow;
class QTransferFunction;
class Scene;
struct ViewerState;

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
  GLView3D(QCamera* cam, QTransferFunction* tran, RenderSettings* rs, QWidget* parent = 0);

  /// Destructor.
  ~GLView3D();

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

  void onNewImage(Scene* scene);

  const CCamera& getCamera() { return m_CCamera; }

  void fromViewerState(const ViewerState& s);

signals:
  void ChangedRenderer();

public slots:

  void OnUpdateCamera();
  void OnUpdateTransferFunction(void);
  void OnUpdateRenderer(int);

public:
  CStatus* getStatus();

protected:
  /// Set up GL context and subsidiary objects.
  void initializeGL();

  /// Render the scene with the current view settings.
  void paintGL();

  /// Resize the view.
  void resizeGL(int w, int h);

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
  QTransferFunction* m_transferFunction;

  /// Rendering timer.
  QElapsedTimer m_etimer;

  /// Last mouse position.
  QPoint m_lastPos;

  RenderSettings* m_renderSettings;

  std::unique_ptr<IRenderWindow> m_renderer;
  int m_rendererType;
};
