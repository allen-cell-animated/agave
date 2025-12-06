#pragma once

#include <memory>

#include "glad/glad.h"

#include "glm.h"
#include "renderlib/CCamera.h"
#include "renderlib/ViewerWindow.h"
#include "renderlib/gesture/gesture.h"

#include <QOpenGLWidget>
#include <QTimer>

class AppearanceDataObject;
class CameraDataObject;
class CStatus;
class ImageXYZC;
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
  GLView3D(QRenderSettings* qrs, RenderSettings* rs, Scene* scene, QWidget* parent = 0);

  /// Destructor.
  ~GLView3D();

  QSize minimumSizeHint() const;
  QSize sizeHint() const;

  void initCameraFromImage(Scene* scene);
  void retargetCameraForNewVolume(Scene* scene);
  void toggleCameraProjection();
  enum MANIPULATOR_MODE
  {
    NONE,
    ROT,
    TRANS
  };
  void setManipulatorMode(MANIPULATOR_MODE mode);

  void toggleRotateControls();
  void toggleTranslateControls();
  void showRotateControls(bool show);
  void showTranslateControls(bool show);

  void onNewImage(Scene* scene);

  const CCamera& getCamera() { return *m_viewerWindow->m_CCamera; }
  // tied to the above camera.  CCamera must outlive this:
  // CameraObject* getCameraDataObject() { return m_cameraObject; }
  void setCameraObject(CameraObject* cameraObject);
  AppearanceObject* getAppearanceDataObject() { return m_appearanceDataObject; }

  void fromViewerState(const Serialize::ViewerState& s);

  QPixmap capture();
  QImage captureQimage();

  // DANGER this must NOT outlive the GLView3D
  ViewerWindow* borrowRenderer() { return m_viewerWindow; }

  void pauseRenderLoop();
  void restartRenderLoop();

signals:
  void ChangedRenderer();

public slots:

  void OnUpdateQRenderSettings(void);
  void OnUpdateRenderer(int);
  void OnSelectionChanged(SceneObject* so);

public:
  std::shared_ptr<CStatus> getStatus();

  /// Resize the view.
  void resizeGL(int w, int h);
  void FitToScene(float transitionDurationSeconds = 0.0f);

protected:
  /// Set up GL context and subsidiary objects.
  void initializeGL();

  /// Render the scene with the current view settings.
  void paintGL();

  void keyPressEvent(QKeyEvent* event);
  void mousePressEvent(QMouseEvent* event);
  void mouseReleaseEvent(QMouseEvent* event);
  void mouseMoveEvent(QMouseEvent* event);
  void wheelEvent(QWheelEvent* event);

private:
  CameraObject* m_cameraObject;
  AppearanceObject* m_appearanceDataObject;
  QRenderSettings* m_qrendersettings;

  /// Rendering timer.
  QTimer* m_etimer;

  ViewerWindow* m_viewerWindow;

  MANIPULATOR_MODE m_manipulatorMode = MANIPULATOR_MODE::NONE;
};
