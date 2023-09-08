#include "Light.h"

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

  // Determine light position
  m_P.x = m_Distance * sinf(m_Phi) * sinf(m_Theta);
  m_P.z = m_Distance * sinf(m_Phi) * cosf(m_Theta);
  m_P.y = m_Distance * cosf(m_Phi);

  m_P += m_Target;

  // Determine area
  if (m_T == 0) {
    m_Area = m_Width * m_Height;
    m_AreaPdf = 1.0f / m_Area;
  }

  if (m_T == 1) {
    m_P = bbctr;
    // shift by nonzero amount
    m_Target = m_P + glm::vec3(0.0, 0.0, 1.0);
    m_SkyRadius = 1000.0f * glm::length(BoundingBox.GetMaxP() - BoundingBox.GetMinP());
    m_Area = 4.0f * PI_F * powf(m_SkyRadius, 2.0f);
    m_AreaPdf = 1.0f / m_Area;
  }

  // after target and p are set...
  updateBasisFrame();
}

void
Light::updateBasisFrame()
{
  // Compute orthogonal basis frame
  m_N = glm::normalize(m_Target - m_P);
  // if N and "up" are parallel, then just choose a different "up"
  if (m_N.y == 1.0f || m_N.y == -1.0f) {
    m_U = glm::normalize(glm::cross(m_N, glm::vec3(1.0f, 0.0f, 0.0f)));
  } else {
    // standard "up" vector
    m_U = glm::normalize(glm::cross(m_N, glm::vec3(0.0f, 1.0f, 0.0f)));
  }
  m_V = glm::normalize(glm::cross(m_N, m_U));
}
