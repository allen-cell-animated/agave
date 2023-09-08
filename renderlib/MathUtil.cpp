#include "MathUtil.h"

float
LinearSpace3f::determinant() const
{
  return dot(vx, cross(vy, vz));
}

LinearSpace3f
LinearSpace3f::transpose() const
{
  return LinearSpace3f({ vx.x, vy.x, vz.x }, { vx.y, vy.y, vz.y }, { vx.z, vy.z, vz.z });
}

LinearSpace3f
LinearSpace3f::adjoint() const
{
  return LinearSpace3f(cross(vy, vz), cross(vz, vx), cross(vx, vy)).transpose();
}

LinearSpace3f
LinearSpace3f::inverse() const
{
  const float det = determinant();
  const struct LinearSpace3f adj = adjoint();
  return LinearSpace3f(adj.vx / det, adj.vy / det, adj.vz / det);
}

AffineSpace3f
AffineSpace3f::inverse() const
{
  const struct LinearSpace3f il = l.inverse();
  glm::vec3 ip = -xfmVector(il, p);
  return AffineSpace3f(il, ip);
}

AffineSpace3f::AffineSpace3f(const glm::quat& orientation, const glm::vec3& p)
  : p(p)
{
  // convert quat to linspace:
  glm::mat3 m = glm::mat3_cast(orientation);
  l = LinearSpace3f(m[0], m[1], m[2]);
}
