#include "BoundingBox.h"
#include "glm.h"

#include <array>

CRegion
CBoundingBox::projectToXY(const glm::mat4& transform) const
{
  std::array<glm::vec4, 8> corners;
  corners[0] = glm::vec4(m_MinP.x, m_MinP.y, m_MinP.z, 1.0f);
  corners[1] = glm::vec4(m_MaxP.x, m_MinP.y, m_MinP.z, 1.0f);
  corners[2] = glm::vec4(m_MinP.x, m_MaxP.y, m_MinP.z, 1.0f);
  corners[3] = glm::vec4(m_MaxP.x, m_MaxP.y, m_MinP.z, 1.0f);
  corners[4] = glm::vec4(m_MinP.x, m_MinP.y, m_MaxP.z, 1.0f);
  corners[5] = glm::vec4(m_MaxP.x, m_MinP.y, m_MaxP.z, 1.0f);
  corners[6] = glm::vec4(m_MinP.x, m_MaxP.y, m_MaxP.z, 1.0f);
  corners[7] = glm::vec4(m_MaxP.x, m_MaxP.y, m_MaxP.z, 1.0f);
  CRegion region;
  for (int i = 0; i < 8; i++) {
    glm::vec4 p = corners[i];
    glm::vec4 p2 = transform * p;
    region.extend(glm::vec2(p2.x / p2.w, p2.y / p2.w));
  }
  return region;
}
