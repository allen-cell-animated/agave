#include "Lighting.h"

Lighting::Lighting(const Lighting& other)
{
  m_NoLights = other.m_NoLights;
  for (int i = 0; i < MAX_NO_LIGHTS; ++i) {
    if (other.m_Lights[i]) {
      m_Lights[i] = new Light(*other.m_Lights[i]);
      m_sceneLights[i] = new SceneLight(m_Lights[i]);
    } else {
      m_Lights[i] = nullptr;
      m_sceneLights[i] = nullptr;
    }
  }
}
