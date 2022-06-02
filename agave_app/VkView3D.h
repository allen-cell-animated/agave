#pragma once

#include <memory>

#include "CameraController.h"
#include "glm.h"
#include "renderlib/CCamera.h"

#include <QElapsedTimer>
#include <QVulkanDeviceFunctions>
#include <QVulkanWindow>
#include <QVulkanWindowRenderer>

class CStatus;
class ImageXYZC;
class QCamera;
class IRenderWindow;
class QRenderSettings;
class Scene;
struct ViewerState;

class VulkanWindowRenderer : public QVulkanWindowRenderer
{
public:
  VulkanWindowRenderer(QVulkanWindow* w)
    : m_window(w)
  {}

  void initResources() override
  {
    m_devFuncs = m_window->vulkanInstance()->deviceFunctions(m_window->device());
    //...
  }
  void initSwapChainResources() override {}
  void releaseSwapChainResources() override {}
  void releaseResources() override {}

  void startNextFrame() override
  {
    // VkCommandBuffer cmdBuf = m_window->currentCommandBuffer();
    // ...
    // m_devFuncs->vkCmdBeginRenderPass(...);
    // ...
    m_window->frameReady();
  }

private:
  QVulkanWindow* m_window;
  QVulkanDeviceFunctions* m_devFuncs;
};

/**
 * 3D GL view of an image with axes and gridlines.
 */
class VkView3D : public QVulkanWindow
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
  VkView3D(QCamera* cam, QRenderSettings* qrs, RenderSettings* rs);

  /// Destructor.
  ~VkView3D();

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

  void fromViewerState(const ViewerState& s);

  QPixmap capture();
  QImage captureQimage();

  QVulkanWindowRenderer* createRenderer() override { return new VulkanWindowRenderer(this); }

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
  QRenderSettings* m_qrendersettings;

  /// Rendering timer.
  QElapsedTimer m_etimer;

  /// Last mouse position.
  QPoint m_lastPos;

  RenderSettings* m_renderSettings;

  std::unique_ptr<IRenderWindow> m_renderer;
  int m_rendererType;
};
