#pragma once

#include <vulkan/vulkan.h>
#include <QWindow>
#include <QVulkanInstance>

class QVulkanDeviceFunctions;
class QVulkanFunctions;

class GestureGraphicsVK
{
public:
  GestureGraphicsVK();
  ~GestureGraphicsVK();

  void init(VkInstance instance, VkDevice device, VkPhysicalDevice physicalDevice,
            VkCommandPool commandPool, VkQueue graphicsQueue);
  void cleanup();

  // Gesture handling
  void startPan(float x, float y);
  void updatePan(float x, float y);
  void endPan();

  void startZoom(float x, float y, float scale);
  void updateZoom(float x, float y, float scale);
  void endZoom();

  void startRotate(float x, float y);
  void updateRotate(float x, float y);
  void endRotate();

  // Camera transformations
  void applyGestures();
  void resetGestures();

  // Get transformation matrices
  const float* getViewMatrix() const { return m_viewMatrix; }
  const float* getProjectionMatrix() const { return m_projectionMatrix; }
  const float* getModelMatrix() const { return m_modelMatrix; }

  // Camera properties
  void setFieldOfView(float fov);
  void setAspectRatio(float aspect);
  void setNearPlane(float nearPlane);
  void setFarPlane(float farPlane);

  // View parameters
  void setViewport(uint32_t width, uint32_t height);
  void setCameraPosition(float x, float y, float z);
  void setCameraTarget(float x, float y, float z);
  void setCameraUp(float x, float y, float z);

private:
  // Vulkan objects
  VkInstance m_instance;
  VkDevice m_device;
  VkPhysicalDevice m_physicalDevice;
  VkCommandPool m_commandPool;
  VkQueue m_graphicsQueue;

  // Gesture state
  bool m_panning;
  bool m_zooming;
  bool m_rotating;
  
  float m_lastPanX, m_lastPanY;
  float m_lastZoomX, m_lastZoomY, m_lastZoomScale;
  float m_lastRotateX, m_lastRotateY;

  // Camera state
  float m_cameraX, m_cameraY, m_cameraZ;
  float m_targetX, m_targetY, m_targetZ;
  float m_upX, m_upY, m_upZ;

  float m_fieldOfView;
  float m_aspectRatio;
  float m_nearPlane;
  float m_farPlane;

  uint32_t m_viewportWidth;
  uint32_t m_viewportHeight;

  // Transform matrices (column-major)
  float m_viewMatrix[16];
  float m_projectionMatrix[16];
  float m_modelMatrix[16];

  // Gesture accumulation
  float m_panDeltaX, m_panDeltaY;
  float m_zoomFactor;
  float m_rotationX, m_rotationY;

  // Helper functions
  void calculateViewMatrix();
  void calculateProjectionMatrix();
  void calculateModelMatrix();
  void multiplyMatrix(const float* a, const float* b, float* result);
  void identityMatrix(float* matrix);
  void translationMatrix(float x, float y, float z, float* matrix);
  void rotationMatrix(float angleX, float angleY, float angleZ, float* matrix);
  void scaleMatrix(float scale, float* matrix);
  void perspectiveMatrix(float fov, float aspect, float nearZ, float farZ, float* matrix);
  void lookAtMatrix(float eyeX, float eyeY, float eyeZ,
                   float centerX, float centerY, float centerZ,
                   float upX, float upY, float upZ, float* matrix);
};