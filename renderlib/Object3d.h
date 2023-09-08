#pragma once

#include "MathUtil.h"

// an object that can be transformed in 3d space
class Transform3d
{
public:
  Transform3d() {}
  virtual ~Transform3d() {}

  AffineSpace3f getAffineSpace() const { return AffineSpace3f(m_rotation, m_center); }
  glm::mat4 getMatrix() const { return glm::translate(glm::mat4_cast(m_rotation), m_center); }

  void applyTranslation(const glm::vec3& translation) { m_center += translation; }
  void applyRotation(const glm::quat& rotation) { m_rotation = rotation * m_rotation; }

  glm::vec3 m_center;
  glm::quat m_rotation;
  // no scaling yet
};
