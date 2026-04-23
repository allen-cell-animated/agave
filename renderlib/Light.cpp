#include "Light.h"

#include "CCamera.h"

void
Light::Update(const CBoundingBox& BoundingBox)
{
  m_InvWidth = 1.0f / m_Width;
  m_HalfWidth = 0.5f * m_Width;
  m_InvHalfWidth = 1.0f / m_HalfWidth;
  m_InvHeight = 1.0f / m_Height;
  m_HalfHeight = 0.5f * m_Height;
  m_InvHalfHeight = 1.0f / m_HalfHeight;
  glm::vec3 bbctr = BoundingBox.GetCenter();
  m_Target = bbctr;

  // Determine light direction away from target from angles
  glm::vec3 dir;
  sphericalToCartesian(m_Phi, m_Theta, dir);
  m_P = m_Target + m_Distance * dir;

  // Determine area for area light
  if (m_T == 0) {
    m_Area = m_Width * m_Height;
    m_AreaPdf = 1.0f / m_Area;
  }

  // Determine area for sky light
  if (m_T == 1) {
    m_Target = bbctr;
    // point on unit sphere around target in direction of spherical angles
    m_P = m_Target + dir;
    m_SkyRadius = 1000.0f * glm::length(BoundingBox.GetMaxP() - BoundingBox.GetMinP());
    m_Area = 4.0f * PI_F * powf(m_SkyRadius, 2.0f);
    m_AreaPdf = 1.0f / m_Area;
  }

  // after target and p are set...
  updateBasisFrame();
}

void
Light::resetArea()
{
  m_Theta = 0.0f;
  m_Phi = HALF_PI_F;
  m_Width = 0.15f;
  m_Height = 0.15f;
  m_Distance = 1.5f;
  m_Color = 10.0f * glm::vec3(1.0f, 1.0f, 1.0f);
  m_ColorIntensity = 1.0f;
}

void
Light::resetSphere()
{
  m_ColorTop = glm::vec3(0.5f, 0.5f, 0.5f);
  m_ColorTopIntensity = 1.0f;
  m_ColorMiddle = glm::vec3(0.5f, 0.5f, 0.5f);
  m_ColorMiddleIntensity = 1.0f;
  m_ColorBottom = glm::vec3(0.5f, 0.5f, 0.5f);
  m_ColorBottomIntensity = 1.0f;
}

void
Light::updateBasisFrame()
{
  // N points from target to light position for sphere/sky lights, and from
  // light to target for area lights.
  const glm::vec3 dir = (m_T == LightType_Sphere) ? (m_P - m_Target) : (m_Target - m_P);
  const glm::mat3 frame = buildOrthonormalFrame(dir, m_U);
  m_U = frame[0];
  m_V = frame[1];
  m_N = frame[2];
}

void
Light::sphericalToCartesian(float phi, float theta, glm::vec3& v)
{
  v.x = sinf(phi) * sinf(theta);
  v.z = sinf(phi) * cosf(theta);
  v.y = cosf(phi);
}

void
Light::sphericalToQuaternion(float phi, float theta, glm::quat& q)
{
  // Convert spherical angles to Cartesian direction
  glm::vec3 dir;
  Light::sphericalToCartesian(phi, theta, dir);

  // Compute the rotation quaternion that rotates from default direction (0,0,1) to dir
  glm::vec3 defaultDir(0.0f, 0.0f, 1.0f);
  float dot = glm::dot(defaultDir, dir);
  if (dot < -0.999999f) {
    // 180 degree rotation around any perpendicular axis
    q = glm::angleAxis(glm::pi<float>(), glm::vec3(1.0f, 0.0f, 0.0f));
  } else {
    // Standard quaternion from two vectors
    glm::vec3 axis = glm::cross(defaultDir, dir);
    q = glm::quat(1.0f + dot, axis.x, axis.y, axis.z);
    q = glm::normalize(q);
  }
}

void
Light::cartesianToSpherical(glm::vec3 v, float& phi, float& theta)
{
  phi = acosf(v.y);
  theta = atan2f(v.x, v.z);
}
