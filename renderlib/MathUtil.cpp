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

// physicalscale is max of physical dims x,y,z
float
computePhysicalScaleBarSize(const float physicalScale)
{
  // note this result will always be some integer power of 10 independent of zoom...
  return pow(10.0f, floor(log10(physicalScale / 2.0f)));
}

Plane
Plane::transform(const glm::mat4& m) const
{
  glm::vec4 O = glm::vec4(normal * -d, 1);
  glm::vec4 N = glm::vec4(normal, 0);
  O = m * O;
  N = glm::normalize(m * N); // use inverse transpose for normals
  return Plane(glm::vec3(N), glm::vec3(O));
}
