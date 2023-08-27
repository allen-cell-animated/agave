#pragma once

#include "glm.h"

struct LinearSpace3f
{
  glm::vec3 vx;
  glm::vec3 vy;
  glm::vec3 vz;

  LinearSpace3f()
    : vx(1.0f, 0.0f, 0.0f)
    , vy(0.0f, 1.0f, 0.0f)
    , vz(0.0f, 0.0f, 1.0f)
  {
  }
  LinearSpace3f(glm::vec3 vx, glm::vec3 vy, glm::vec3 vz)
    : vx(vx)
    , vy(vy)
    , vz(vz)
  {
  }

  LinearSpace3f inverse() const;

private:
  float determinant() const;
  LinearSpace3f transpose() const;
  LinearSpace3f adjoint() const;
};

struct AffineSpace3f
{
public:
  struct LinearSpace3f l;
  glm::vec3 p;

  AffineSpace3f()
    : l()
    , p()
  {
  }
  AffineSpace3f(struct LinearSpace3f l, glm::vec3 p)
    : l(l)
    , p(p)
  {
  }
  AffineSpace3f inverse() const;
};

inline glm::vec3
xfmVector(const struct LinearSpace3f& xfm, glm::vec3 p)
{
  return xfm.vx * p.x + xfm.vy * p.y + xfm.vz * p.z;
}
inline glm::vec3
xfmVector(const struct AffineSpace3f& xfm, glm::vec3 p)
{
  return xfmVector(xfm.l, p) + xfm.p;
}

/// @brief Intersect a line with a plane and return the parametric distance along the
/// line. The plane is expressed in vector form such as for any point P on the plane
/// this condition is true:
///     dot((P - planeP), planeN) = 0
/// The line is epressed in parametric form:
///     P = lineP + lineV * lineT
///
/// This function solves for distance lineT.
inline float
linePlaneIsectT(const glm::vec3& planeP, const glm::vec3& planeN, const glm::vec3& lineP, const glm::vec3& lineV)
{
  float det = dot(lineV, planeN);
  float lineT = (abs(det) < 1e-19f ? 0.0f : dot(planeP - lineP, planeN) / det);

  return lineT;
}

/// @brief Compute the intersection point of a line and a plane
inline glm::vec3
linePlaneIsect(const glm::vec3& planeP, const glm::vec3& planeN, const glm::vec3& lineP, const glm::vec3& lineV)
{
  float lineT = linePlaneIsectT(planeP, planeN, lineP, lineV);
  glm::vec3 isectP = lineP + lineV * lineT;
  return isectP;
}

/// @brief Compute the nearest point to line B along line A.
inline glm::vec3
lineLineNearestPoint(const glm::vec3& lineA_P,
                     const glm::vec3 lineA_V,
                     const glm::vec3& lineB_P,
                     const glm::vec3 lineB_V)
{
  // Compute direction of the line perpendicular to the two lines.
  glm::vec3 perpV = cross(lineA_V, lineB_V);

  // Compute the normal of the plane formed between lineB and the perpendicular.
  glm::vec3 N = cross(perpV, lineB_V);

  // The point on lineA is at the intersection between the line and plane
  // lineB_P -> N
  return linePlaneIsect(lineB_P, N, lineA_P, lineA_V);
}

inline float
distanceToPlane(const glm::vec3& p, const glm::vec3& planeP, const glm::vec3& planeN)
{
  return glm::dot(p - planeP, planeN);
}

inline glm::vec3
circleNearestPoint(const glm::vec3& p, const glm::vec3& center, const glm::vec3& normal, float r)
{
  float d = distanceToPlane(p, center, normal);
  glm::vec3 p2 = p - d * normal;

  // vector from center to point
  glm::vec3 v = p2 - center;
  float magV = glm::length(v);
  // reduce distance to radius and compute pt on circle
  // i.e. compute pt on circle toward direction of p
  glm::vec3 a = center + v * (r / magV);
  return a;
}

inline float
lerp(float a, float b, float alpha)
{
  return a + alpha * (b - a);
}