#pragma once

#include "BoundingBox.h"
#include "Defines.h"
#include "MathUtil.h"
#include "Object3d.h"

#include <functional>
#include <vector>

// this should map to the bundle of gpu parameters
// passed to the shader for our lights
class Light
{
public:
  // range is 0..2pi
  float m_Theta;
  // range is 0..pi
  float m_Phi;
  float m_Width;
  float m_InvWidth;
  float m_HalfWidth;
  float m_InvHalfWidth;
  float m_Height;
  float m_InvHeight;
  float m_HalfHeight;
  float m_InvHalfHeight;
  float m_Distance;
  float m_SkyRadius;
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
  float m_ColorIntensity;
  float m_ColorTopIntensity;
  float m_ColorMiddleIntensity;
  float m_ColorBottomIntensity;
  // 0 for area light, 1 for sky light
  int m_T;

  Light(void)
    : m_Theta(0.0f)
    , m_Phi(HALF_PI_F)
    , m_Width(1.0f)
    , m_InvWidth(1.0f / m_Width)
    , m_HalfWidth(0.5f * m_Width)
    , m_InvHalfWidth(1.0f / m_HalfWidth)
    , m_Height(1.0f)
    , m_InvHeight(1.0f / m_Height)
    , m_HalfHeight(0.5f * m_Height)
    , m_InvHalfHeight(1.0f / m_HalfHeight)
    , m_Distance(1.0f)
    , m_SkyRadius(100.0f)
    , m_P(1.0f, 1.0f, 1.0f)
    , m_Target(0.0f, 0.0f, 0.0f)
    , m_N(1.0f, 0.0f, 0.0f)
    , m_U(1.0f, 0.0f, 0.0f)
    , m_V(1.0f, 0.0f, 0.0f)
    , m_Area(m_Width * m_Height)
    , m_AreaPdf(1.0f / m_Area)
    , m_Color(10.0f)
    , m_ColorTop(10.0f)
    , m_ColorMiddle(10.0f)
    , m_ColorBottom(10.0f)
    , m_ColorIntensity(1.0f)
    , m_ColorTopIntensity(1.0f)
    , m_ColorMiddleIntensity(1.0f)
    , m_ColorBottomIntensity(1.0f)
    , m_T(0)
  {
  }

  Light& operator=(const Light& Other)
  {
    m_Theta = Other.m_Theta;
    m_Phi = Other.m_Phi;
    m_Width = Other.m_Width;
    m_InvWidth = Other.m_InvWidth;
    m_HalfWidth = Other.m_HalfWidth;
    m_InvHalfWidth = Other.m_InvHalfWidth;
    m_Height = Other.m_Height;
    m_InvHeight = Other.m_InvHeight;
    m_HalfHeight = Other.m_HalfHeight;
    m_InvHalfHeight = Other.m_InvHalfHeight;
    m_Distance = Other.m_Distance;
    m_SkyRadius = Other.m_SkyRadius;
    m_P = Other.m_P;
    m_Target = Other.m_Target;
    m_N = Other.m_N;
    m_U = Other.m_U;
    m_V = Other.m_V;
    m_Area = Other.m_Area;
    m_AreaPdf = Other.m_AreaPdf;
    m_Color = Other.m_Color;
    m_ColorTop = Other.m_ColorTop;
    m_ColorMiddle = Other.m_ColorMiddle;
    m_ColorBottom = Other.m_ColorBottom;
    m_ColorIntensity = Other.m_ColorIntensity;
    m_ColorTopIntensity = Other.m_ColorTopIntensity;
    m_ColorMiddleIntensity = Other.m_ColorMiddleIntensity;
    m_ColorBottomIntensity = Other.m_ColorBottomIntensity;
    m_T = Other.m_T;

    return *this;
  }

  void Update(const CBoundingBox& BoundingBox);
  void updateBasisFrame();

  static void sphericalToCartesian(float phi, float theta, glm::vec3& v);
  static void cartesianToSpherical(glm::vec3 v, float& phi, float& theta);
};

// MUST NOT OUTLIVE ITS LIGHT
class SceneLight : public SceneObject
{
public:
  void updateTransform();
  Light* m_light;
  std::vector<std::function<void(const Light&)>> m_observers;
};
