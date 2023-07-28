#pragma once

#include "BoundingBox.h"
#include "Defines.h"
#include "DenoiseParams.h"
#include "GradientData.h"
#include "Timeline.h"
#include "glm.h"

#include <memory>
#include <vector>

class ImageXYZC;
class Manipulator;

#define MAX_CPU_CHANNELS 32
struct VolumeDisplay
{
  bool m_showBoundingBox = true;
  float m_boundingBoxColor[3] = { 1.0f, 1.0f, 1.0f };
  float m_backgroundColor[3] = { 0.0f, 0.0f, 0.0f };
  float m_DensityScale = 50.0f;
  float m_GradientFactor = 0.1f;
  int m_ShadingType = 2;

  // channels enabled/disabled
  // per channel colors
  float m_diffuse[MAX_CPU_CHANNELS * 3];
  float m_specular[MAX_CPU_CHANNELS * 3];
  float m_emissive[MAX_CPU_CHANNELS * 3];
  float m_roughness[MAX_CPU_CHANNELS];
  float m_opacity[MAX_CPU_CHANNELS];
  bool m_enabled[MAX_CPU_CHANNELS];

  GradientData m_gradientData[MAX_CPU_CHANNELS];
};

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
};

#define MAX_NO_LIGHTS 4
class Lighting
{
public:
  Lighting(void)
    : m_NoLights(0)
  {
  }

  Lighting& operator=(const Lighting& Other)
  {
    for (int i = 0; i < MAX_NO_LIGHTS; i++) {
      m_Lights[i] = Other.m_Lights[i];
    }

    m_NoLights = Other.m_NoLights;

    return *this;
  }

  void AddLight(const Light& Light)
  {
    if (m_NoLights >= MAX_NO_LIGHTS)
      return;

    m_Lights[m_NoLights] = Light;

    m_NoLights = m_NoLights + 1;
  }

  void Reset(void)
  {
    m_NoLights = 0;
    // memset(m_Lights, 0 , MAX_NO_LIGHTS * sizeof(CLight));
  }

  Light m_Lights[MAX_NO_LIGHTS];
  int m_NoLights;
};

class Scene
{
public:
  Timeline m_timeLine;

  // one single volume, for now...!
  std::shared_ptr<ImageXYZC> m_volume;
  // appearance settings for a volume
  VolumeDisplay m_material;
  CBoundingBox m_roi = CBoundingBox(glm::vec3(0, 0, 0), glm::vec3(1, 1, 1));

  Lighting m_lighting;

  std::vector<Manipulator*> m_tools;

  CBoundingBox m_boundingBox;

  void initLights();
  void initSceneFromImg(std::shared_ptr<ImageXYZC> img);
  void initBounds(const CBoundingBox& bb);
  void initBoundsFromImg(std::shared_ptr<ImageXYZC> img);
  void getFirst4EnabledChannels(uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3) const;
};
