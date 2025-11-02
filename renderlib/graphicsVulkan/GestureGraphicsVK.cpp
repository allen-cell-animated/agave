#include "GestureGraphicsVK.h"
#include "Logging.h"
#include <cmath>
#include <cstring>

GestureGraphicsVK::GestureGraphicsVK()
  : m_instance(VK_NULL_HANDLE)
  , m_device(VK_NULL_HANDLE)
  , m_physicalDevice(VK_NULL_HANDLE)
  , m_commandPool(VK_NULL_HANDLE)
  , m_graphicsQueue(VK_NULL_HANDLE)
  , m_panning(false)
  , m_zooming(false)
  , m_rotating(false)
  , m_lastPanX(0.0f)
  , m_lastPanY(0.0f)
  , m_lastZoomX(0.0f)
  , m_lastZoomY(0.0f)
  , m_lastZoomScale(1.0f)
  , m_lastRotateX(0.0f)
  , m_lastRotateY(0.0f)
  , m_cameraX(0.0f)
  , m_cameraY(0.0f)
  , m_cameraZ(5.0f)
  , m_targetX(0.0f)
  , m_targetY(0.0f)
  , m_targetZ(0.0f)
  , m_upX(0.0f)
  , m_upY(1.0f)
  , m_upZ(0.0f)
  , m_fieldOfView(45.0f)
  , m_aspectRatio(1.0f)
  , m_nearPlane(0.1f)
  , m_farPlane(100.0f)
  , m_viewportWidth(800)
  , m_viewportHeight(600)
  , m_panDeltaX(0.0f)
  , m_panDeltaY(0.0f)
  , m_zoomFactor(1.0f)
  , m_rotationX(0.0f)
  , m_rotationY(0.0f)
{
  identityMatrix(m_viewMatrix);
  identityMatrix(m_projectionMatrix);
  identityMatrix(m_modelMatrix);
}

GestureGraphicsVK::~GestureGraphicsVK()
{
  cleanup();
}

void
GestureGraphicsVK::init(VkInstance instance,
                        VkDevice device,
                        VkPhysicalDevice physicalDevice,
                        VkCommandPool commandPool,
                        VkQueue graphicsQueue)
{
  m_instance = instance;
  m_device = device;
  m_physicalDevice = physicalDevice;
  m_commandPool = commandPool;
  m_graphicsQueue = graphicsQueue;

  // Calculate initial matrices
  calculateViewMatrix();
  calculateProjectionMatrix();
  calculateModelMatrix();

  LOG_INFO << "GestureGraphicsVK initialized";
}

void
GestureGraphicsVK::cleanup()
{
  m_instance = VK_NULL_HANDLE;
  m_device = VK_NULL_HANDLE;
  m_physicalDevice = VK_NULL_HANDLE;
  m_commandPool = VK_NULL_HANDLE;
  m_graphicsQueue = VK_NULL_HANDLE;
}

void
GestureGraphicsVK::startPan(float x, float y)
{
  m_panning = true;
  m_lastPanX = x;
  m_lastPanY = y;
}

void
GestureGraphicsVK::updatePan(float x, float y)
{
  if (m_panning) {
    float deltaX = x - m_lastPanX;
    float deltaY = y - m_lastPanY;

    // Accumulate pan deltas
    m_panDeltaX += deltaX * 0.01f;
    m_panDeltaY -= deltaY * 0.01f; // Invert Y for proper camera movement

    m_lastPanX = x;
    m_lastPanY = y;

    applyGestures();
  }
}

void
GestureGraphicsVK::endPan()
{
  m_panning = false;
}

void
GestureGraphicsVK::startZoom(float x, float y, float scale)
{
  m_zooming = true;
  m_lastZoomX = x;
  m_lastZoomY = y;
  m_lastZoomScale = scale;
}

void
GestureGraphicsVK::updateZoom(float x, float y, float scale)
{
  if (m_zooming) {
    float deltaScale = scale / m_lastZoomScale;
    m_zoomFactor *= deltaScale;

    // Clamp zoom factor
    m_zoomFactor = std::max(0.1f, std::min(10.0f, m_zoomFactor));

    m_lastZoomX = x;
    m_lastZoomY = y;
    m_lastZoomScale = scale;

    applyGestures();
  }
}

void
GestureGraphicsVK::endZoom()
{
  m_zooming = false;
}

void
GestureGraphicsVK::startRotate(float x, float y)
{
  m_rotating = true;
  m_lastRotateX = x;
  m_lastRotateY = y;
}

void
GestureGraphicsVK::updateRotate(float x, float y)
{
  if (m_rotating) {
    float deltaX = x - m_lastRotateX;
    float deltaY = y - m_lastRotateY;

    // Accumulate rotation deltas
    m_rotationX += deltaY * 0.01f;
    m_rotationY += deltaX * 0.01f;

    m_lastRotateX = x;
    m_lastRotateY = y;

    applyGestures();
  }
}

void
GestureGraphicsVK::endRotate()
{
  m_rotating = false;
}

void
GestureGraphicsVK::applyGestures()
{
  calculateViewMatrix();
  calculateModelMatrix();
}

void
GestureGraphicsVK::resetGestures()
{
  m_panDeltaX = 0.0f;
  m_panDeltaY = 0.0f;
  m_zoomFactor = 1.0f;
  m_rotationX = 0.0f;
  m_rotationY = 0.0f;

  applyGestures();
}

void
GestureGraphicsVK::setFieldOfView(float fov)
{
  m_fieldOfView = fov;
  calculateProjectionMatrix();
}

void
GestureGraphicsVK::setAspectRatio(float aspect)
{
  m_aspectRatio = aspect;
  calculateProjectionMatrix();
}

void
GestureGraphicsVK::setNearPlane(float nearPlane)
{
  m_nearPlane = nearPlane;
  calculateProjectionMatrix();
}

void
GestureGraphicsVK::setFarPlane(float farPlane)
{
  m_farPlane = farPlane;
  calculateProjectionMatrix();
}

void
GestureGraphicsVK::setViewport(uint32_t width, uint32_t height)
{
  m_viewportWidth = width;
  m_viewportHeight = height;
  m_aspectRatio = static_cast<float>(width) / static_cast<float>(height);
  calculateProjectionMatrix();
}

void
GestureGraphicsVK::setCameraPosition(float x, float y, float z)
{
  m_cameraX = x;
  m_cameraY = y;
  m_cameraZ = z;
  calculateViewMatrix();
}

void
GestureGraphicsVK::setCameraTarget(float x, float y, float z)
{
  m_targetX = x;
  m_targetY = y;
  m_targetZ = z;
  calculateViewMatrix();
}

void
GestureGraphicsVK::setCameraUp(float x, float y, float z)
{
  m_upX = x;
  m_upY = y;
  m_upZ = z;
  calculateViewMatrix();
}

void
GestureGraphicsVK::calculateViewMatrix()
{
  // Apply pan offsets to camera position
  float eyeX = m_cameraX + m_panDeltaX;
  float eyeY = m_cameraY + m_panDeltaY;
  float eyeZ = m_cameraZ / m_zoomFactor;

  lookAtMatrix(eyeX, eyeY, eyeZ, m_targetX, m_targetY, m_targetZ, m_upX, m_upY, m_upZ, m_viewMatrix);
}

void
GestureGraphicsVK::calculateProjectionMatrix()
{
  perspectiveMatrix(m_fieldOfView, m_aspectRatio, m_nearPlane, m_farPlane, m_projectionMatrix);
}

void
GestureGraphicsVK::calculateModelMatrix()
{
  float rotMatrix[16];
  float scaleMatrix[16];
  float tempMatrix[16];

  // Start with identity
  identityMatrix(m_modelMatrix);

  // Apply rotation
  rotationMatrix(m_rotationX, m_rotationY, 0.0f, rotMatrix);
  multiplyMatrix(m_modelMatrix, rotMatrix, tempMatrix);
  memcpy(m_modelMatrix, tempMatrix, sizeof(m_modelMatrix));

  // Apply scale (zoom is handled by camera distance)
  this->scaleMatrix(1.0f, scaleMatrix);
  multiplyMatrix(m_modelMatrix, scaleMatrix, tempMatrix);
  memcpy(m_modelMatrix, tempMatrix, sizeof(m_modelMatrix));
}

void
GestureGraphicsVK::multiplyMatrix(const float* a, const float* b, float* result)
{
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      result[i * 4 + j] = 0.0f;
      for (int k = 0; k < 4; k++) {
        result[i * 4 + j] += a[i * 4 + k] * b[k * 4 + j];
      }
    }
  }
}

void
GestureGraphicsVK::identityMatrix(float* matrix)
{
  memset(matrix, 0, 16 * sizeof(float));
  matrix[0] = matrix[5] = matrix[10] = matrix[15] = 1.0f;
}

void
GestureGraphicsVK::translationMatrix(float x, float y, float z, float* matrix)
{
  identityMatrix(matrix);
  matrix[12] = x;
  matrix[13] = y;
  matrix[14] = z;
}

void
GestureGraphicsVK::rotationMatrix(float angleX, float angleY, float angleZ, float* matrix)
{
  float cosX = cosf(angleX), sinX = sinf(angleX);
  float cosY = cosf(angleY), sinY = sinf(angleY);
  float cosZ = cosf(angleZ), sinZ = sinf(angleZ);

  identityMatrix(matrix);

  // Combined rotation matrix (ZYX order)
  matrix[0] = cosY * cosZ;
  matrix[1] = cosY * sinZ;
  matrix[2] = -sinY;
  matrix[4] = sinX * sinY * cosZ - cosX * sinZ;
  matrix[5] = sinX * sinY * sinZ + cosX * cosZ;
  matrix[6] = sinX * cosY;
  matrix[8] = cosX * sinY * cosZ + sinX * sinZ;
  matrix[9] = cosX * sinY * sinZ - sinX * cosZ;
  matrix[10] = cosX * cosY;
}

void
GestureGraphicsVK::scaleMatrix(float scale, float* matrix)
{
  identityMatrix(matrix);
  matrix[0] = matrix[5] = matrix[10] = scale;
}

void
GestureGraphicsVK::perspectiveMatrix(float fov, float aspect, float nearZ, float farZ, float* matrix)
{
  float f = 1.0f / tanf(fov * M_PI / 360.0f);

  memset(matrix, 0, 16 * sizeof(float));
  matrix[0] = f / aspect;
  matrix[5] = f;
  matrix[10] = (farZ + nearZ) / (nearZ - farZ);
  matrix[11] = -1.0f;
  matrix[14] = (2.0f * farZ * nearZ) / (nearZ - farZ);
}

void
GestureGraphicsVK::lookAtMatrix(float eyeX,
                                float eyeY,
                                float eyeZ,
                                float centerX,
                                float centerY,
                                float centerZ,
                                float upX,
                                float upY,
                                float upZ,
                                float* matrix)
{
  // Calculate forward vector (eye to center)
  float fx = centerX - eyeX;
  float fy = centerY - eyeY;
  float fz = centerZ - eyeZ;

  // Normalize forward vector
  float flen = sqrtf(fx * fx + fy * fy + fz * fz);
  if (flen > 0.0f) {
    fx /= flen;
    fy /= flen;
    fz /= flen;
  }

  // Calculate right vector (forward x up)
  float rx = fy * upZ - fz * upY;
  float ry = fz * upX - fx * upZ;
  float rz = fx * upY - fy * upX;

  // Normalize right vector
  float rlen = sqrtf(rx * rx + ry * ry + rz * rz);
  if (rlen > 0.0f) {
    rx /= rlen;
    ry /= rlen;
    rz /= rlen;
  }

  // Calculate new up vector (right x forward)
  float ux = ry * fz - rz * fy;
  float uy = rz * fx - rx * fz;
  float uz = rx * fy - ry * fx;

  // Build matrix
  matrix[0] = rx;
  matrix[4] = ux;
  matrix[8] = -fx;
  matrix[12] = 0.0f;
  matrix[1] = ry;
  matrix[5] = uy;
  matrix[9] = -fy;
  matrix[13] = 0.0f;
  matrix[2] = rz;
  matrix[6] = uz;
  matrix[10] = -fz;
  matrix[14] = 0.0f;
  matrix[3] = 0.0f;
  matrix[7] = 0.0f;
  matrix[11] = 0.0f;
  matrix[15] = 1.0f;

  // Apply translation
  matrix[12] = -(rx * eyeX + ry * eyeY + rz * eyeZ);
  matrix[13] = -(ux * eyeX + uy * eyeY + uz * eyeZ);
  matrix[14] = -(-fx * eyeX + -fy * eyeY + -fz * eyeZ);
}