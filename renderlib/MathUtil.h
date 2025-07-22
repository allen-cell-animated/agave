#pragma once

#include "glm.h"

// see https://github.com/embree/embree/blob/master/common/math/affinespace.h
// Much of this could probably be replaced by just using glm::mat3 and glm::mat4
// but this makes explicit the split between basis vectors and affine component,
// and thereby makes some operations more readable.
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
  AffineSpace3f(struct LinearSpace3f l, const glm::vec3& p)
    : l(l)
    , p(p)
  {
  }
  AffineSpace3f(const glm::quat& orientation, const glm::vec3& p);

  AffineSpace3f inverse() const;
};

// an object that can be transformed in 3d space
class Transform3d
{
public:
  Transform3d()
  {
    m_rotation = glm::quat(glm::vec3(0, 0, 0));
    m_center = glm::vec3(0, 0, 0);
  }
  virtual ~Transform3d() {}

  AffineSpace3f getAffineSpace() const { return AffineSpace3f(m_rotation, m_center); }
  glm::mat4 getMatrix() const;

  void applyTranslation(const glm::vec3& translation) { m_center += translation; }
  void applyRotation(const glm::quat& rotation) { m_rotation = rotation * m_rotation; }

  glm::vec3 m_center;
  glm::quat m_rotation;
  // no scaling yet
};

inline glm::vec3
xfmVector(const struct LinearSpace3f& xfm, const glm::vec3& p)
{
  return xfm.vx * p.x + xfm.vy * p.y + xfm.vz * p.z;
}

// transforming as a point is different than as a vector
inline glm::vec3
xfmPoint(const struct AffineSpace3f& xfm, const glm::vec3& p)
{
  return xfmVector(xfm.l, p) + xfm.p;
}

// transforming as a vector (spatial direction) can ignore the affine p component
inline glm::vec3
xfmVector(const struct AffineSpace3f& xfm, const glm::vec3& p)
{
  return xfmVector(xfm.l, p);
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
lerp(float a, float b, float alpha)
{
  return a + alpha * (b - a);
}

inline glm::quat
trackball(float xRadians, float yRadians, const glm::vec3& eye, const glm::vec3& up, const glm::vec3& right)
{
  float angle = sqrtf(yRadians * yRadians + xRadians * xRadians);
  if (angle == 0.0f) {
    // skip some extra math
    return glm::quat(glm::vec3(0, 0, 0));
  }

  glm::vec3 objectUpDirection = up; // or m_V; ???
  glm::vec3 objectSidewaysDirection = right;

  // negating/inverting these has the effect of tumbling the target and not moving the camera.
  objectUpDirection *= yRadians;
  objectSidewaysDirection *= xRadians;

  glm::vec3 moveDirection = objectUpDirection + objectSidewaysDirection;

  glm::vec3 axis = glm::normalize(glm::cross(moveDirection, eye));

  return glm::angleAxis(angle, axis);
}

float
computePhysicalScaleBarSize(const float physicalScale);

struct Ray
{
  glm::vec3 origin;
  glm::vec3 direction;
  Ray(const glm::vec3& o, const glm::vec3& d)
    : origin(o)
    , direction(d)
  {
  }
};

struct Plane
{
  glm::vec3 normal;
  float d;

  Plane()
    : normal(0.0f, 0.0f, 1.0f)
    , d(0.0f)
  {
    // default plane points to +z and sits at origin in xy plane.
  }

  Plane(const glm::vec3& n, float dist)
    : normal(glm::normalize(n))
    , d(dist)
  {
  }
  Plane(const glm::vec3& n, const glm::vec3& p)
    : normal(glm::normalize(n))
    , d(glm::dot(normal, p)) {};

  // the vec4 version satisfies the plane equation: dot(normal, p) = d but is of the form v0*x + v1*y + v2*z + v3 = 0
  // so you can do dot(asVec4, vec4(p,1)) = 0
  glm::vec4 asVec4() const { return glm::vec4(normal, -d); }

  glm::vec3 getPointInPlane() const { return normal * d; }
  bool isInPlane(const glm::vec3& p, float epsilon = 0.0001f) const { return abs(glm::dot(normal, p) - d) < epsilon; }

  Plane transform(const glm::mat4& m) const;
  Plane transform(const Transform3d& transform) const;

  Transform3d getTransformTo(const Plane& p) const;

  // Credit: https://stackoverflow.com/a/32410473/2373034
  // Returns the intersection line of the 2 planes
  static Ray GetIntersection(const Plane& p1, const Plane& p2)
  {
    glm::vec3 p3Normal = glm::cross(p1.normal, p2.normal);
    float det = glm::length2(p3Normal);

    return Ray(((glm::cross(p3Normal, p2.normal) * -p1.d) + (glm::cross(p1.normal, p3Normal) * -p2.d)) / det, p3Normal);
  }
};

// convert an integer to a float in the range [0,1]
// for uint16, this will return [0, 65535] -> [0.0, 1.0]
// for uint8, this will return [0, 255] -> [0.0, 1.0]
template<typename INTTYPE>
float
normalizeInt(INTTYPE value)
{
  return static_cast<float>(value) / static_cast<float>(std::numeric_limits<INTTYPE>::max());
}
