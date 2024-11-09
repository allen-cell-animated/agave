#pragma once

#include "BoundingBox.h"
#include "Defines.h"
#include "DenoiseParams.h"
#include "GradientData.h"
#include "Light.h"
#include "Object3d.h"
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

  VolumeDisplay();
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
      m_sceneLights[i].m_light = &m_Lights[i];
    }

    m_NoLights = Other.m_NoLights;

    return *this;
  }

  void AddLight(const Light& Light)
  {
    if (m_NoLights >= MAX_NO_LIGHTS)
      return;

    m_Lights[m_NoLights] = Light;
    m_sceneLights[m_NoLights].m_light = &m_Lights[m_NoLights];

    m_NoLights = m_NoLights + 1;
  }

  void Reset(void)
  {
    m_NoLights = 0;
    // memset(m_Lights, 0 , MAX_NO_LIGHTS * sizeof(CLight));
  }

  Light m_Lights[MAX_NO_LIGHTS];
  int m_NoLights;
  SceneLight m_sceneLights[MAX_NO_LIGHTS];
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
  Plane m_userClipPlane;

  Lighting m_lighting;

  std::vector<Manipulator*> m_tools;

  CBoundingBox m_boundingBox;
  bool m_showScaleBar = false;
  bool m_showAxisHelper = false;

  void initLights();
  void initSceneFromImg(std::shared_ptr<ImageXYZC> img);
  void initBounds(const CBoundingBox& bb);
  void initBoundsFromImg(std::shared_ptr<ImageXYZC> img);
  void getFirst4EnabledChannels(uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3) const;
};
