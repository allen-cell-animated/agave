#pragma once

#include "BoundingBox.h"
#include "Defines.h"
#include "MathUtil.h"
#include "Object3d.h"

#include <functional>
#include <vector>
#include <memory>

class CCamera;

static constexpr int LightType_Area = 0;
static constexpr int LightType_Sphere = 1;

// this should map to the bundle of gpu parameters
// passed to the shader for our lights
class Light
{
public:
  // range is 0..2pi
  float m_Theta{ 0.0f };
  // range is 0..pi
  float m_Phi{ HALF_PI_F };
  float m_Width{ 1.0f };
  float m_InvWidth;
  float m_HalfWidth;
  float m_InvHalfWidth;
  float m_Height{ 1.0f };
  float m_InvHeight;
  float m_HalfHeight;
  float m_InvHalfHeight;
  float m_Distance{ 1.0f };
  float m_SkyRadius{ 100.0f };
  glm::vec3 m_P;
  glm::vec3 m_Target;
  glm::vec3 m_N;
  glm::vec3 m_U;
  glm::vec3 m_V;
  float m_Area;
  float m_AreaPdf;
  glm::vec3 m_Color;
  glm::vec3 m_ColorTop;
  glm::vec3 m_ColorMiddle;
  glm::vec3 m_ColorBottom;
  float m_ColorIntensity{ 1.0f };
  float m_ColorTopIntensity{ 1.0f };
  float m_ColorMiddleIntensity{ 1.0f };
  float m_ColorBottomIntensity{ 1.0f };
  // 0 for area light, 1 for sky light
  int m_T{ 0 };

  Light()
    : m_InvWidth(1.0f / m_Width)
    , m_HalfWidth(0.5f * m_Width)
    , m_InvHalfWidth(1.0f / m_HalfWidth)
    , m_InvHeight(1.0f / m_Height)
    , m_HalfHeight(0.5f * m_Height)
    , m_InvHalfHeight(1.0f / m_HalfHeight)
    , m_P(1.0f, 1.0f, 1.0f)
    , m_Target(0.0f, 0.0f, 0.0f)
    , m_N(0.0f, 0.0f, 1.0f)
    , m_U(1.0f, 0.0f, 0.0f)
    , m_V(0.0f, 1.0f, 0.0f)
    , m_Area(m_Width * m_Height)
    , m_AreaPdf(1.0f / m_Area)
    , m_Color(10.0f)
    , m_ColorTop(10.0f)
    , m_ColorMiddle(10.0f)
    , m_ColorBottom(10.0f)
  {
  }

  Light& operator=(const Light& Other) = default;

  void Update(const CBoundingBox& BoundingBox);
  void updateBasisFrame();

  void validateBasis(const char* loglabel) const;

  static void sphericalToCartesian(float phi, float theta, glm::vec3& v);
  static void cartesianToSpherical(glm::vec3 v, float& phi, float& theta);

  static void sphericalToQuaternion(float phi, float theta, glm::quat& q);
};
