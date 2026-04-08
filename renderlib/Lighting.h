#pragma once

#include "Light.h"
#include "SceneLight.h"

static constexpr size_t MAX_NO_LIGHTS = 4;

class Lighting
{
public:
  Lighting() = default;
  ~Lighting()
  {
    for (int i = 0; i < MAX_NO_LIGHTS; ++i) {
      delete m_sceneLights[i];
      delete m_Lights[i];
    }
  }

  Lighting(const Lighting& other);

  Light& LightRef(int i) const { return *m_Lights[i]; }

  void AddLight(Light& light)
  {
    if (m_NoLights >= MAX_NO_LIGHTS)
      return;

    m_Lights[m_NoLights] = new Light(light);
    m_sceneLights[m_NoLights] = new SceneLight(m_Lights[m_NoLights]);

    m_NoLights = m_NoLights + 1;
  }
  void SetLight(int i, Light& light)
  {
    if (i >= MAX_NO_LIGHTS)
      return;

    delete m_sceneLights[i];
    delete m_Lights[i];

    m_Lights[i] = new Light(light);
    m_sceneLights[i] = new SceneLight(m_Lights[i]);
  }

  Light* m_Lights[MAX_NO_LIGHTS]{ nullptr, nullptr, nullptr, nullptr };
  int m_NoLights{ 0 };
  SceneLight* m_sceneLights[MAX_NO_LIGHTS]{ nullptr, nullptr, nullptr, nullptr };

  bool lockToCamera = false;
};
