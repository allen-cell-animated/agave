#pragma once

#include "Light.h"
#include "SceneLight.h"
#include "glm.h"

class CCamera;
class RenderSettings;

static constexpr size_t MAX_NO_LIGHTS = 4;

// Describes the lighting in the scene,
// which is a collection of lights and their properties,
// as well as some utility functions for managing them
class Lighting
{
public:
  Lighting() = default;
  ~Lighting()
  {
    for (size_t i = 0; i < MAX_NO_LIGHTS; ++i) {
      delete m_sceneLights[i];
      delete m_Lights[i];
    }
  }

  Lighting(const Lighting& other);
  Lighting& operator=(const Lighting&) = delete;
  Lighting(Lighting&&) = delete;
  Lighting& operator=(Lighting&&) = delete;

  Light& LightRef(int i) { return *m_Lights[i]; }
  const Light& LightRef(int i) const { return *m_Lights[i]; }

  void AddLight(Light& light)
  {
    if (m_NoLights >= MAX_NO_LIGHTS)
      return;

    m_Lights[m_NoLights] = new Light(light);
    m_sceneLights[m_NoLights] = new SceneLight(m_Lights[m_NoLights]);

    m_NoLights = m_NoLights + 1;
  }
  void SetLight(size_t i, Light& light)
  {
    if (i >= MAX_NO_LIGHTS)
      return;

    delete m_sceneLights[i];
    delete m_Lights[i];

    m_Lights[i] = new Light(light);
    m_sceneLights[i] = new SceneLight(m_Lights[i]);
  }

  void captureLightsViewSpaceBasis(const CCamera& camera);
  void restoreLightsViewSpaceBasis(const CCamera& camera, RenderSettings* rs);

  Light* m_Lights[MAX_NO_LIGHTS]{ nullptr, nullptr, nullptr, nullptr };
  size_t m_NoLights{ 0 };
  SceneLight* m_sceneLights[MAX_NO_LIGHTS]{ nullptr, nullptr, nullptr, nullptr };

  bool lockToCamera = false;
  glm::mat3 m_capturedRelativeBasis[MAX_NO_LIGHTS] = { glm::mat3(1.0f),
                                                       glm::mat3(1.0f),
                                                       glm::mat3(1.0f),
                                                       glm::mat3(1.0f) };
};
