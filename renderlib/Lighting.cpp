#include "Lighting.h"

#include "CCamera.h"
#include "Enumerations.h"
#include "RenderSettings.h"

Lighting::Lighting(const Lighting& other)
{
  m_NoLights = other.m_NoLights;
  lockToCamera = other.lockToCamera;
  for (size_t i = 0; i < MAX_NO_LIGHTS; ++i) {
    m_capturedRelativeBasis[i] = other.m_capturedRelativeBasis[i];
    if (other.m_Lights[i]) {
      m_Lights[i] = new Light(*other.m_Lights[i]);
      m_sceneLights[i] = new SceneLight(m_Lights[i]);
    } else {
      m_Lights[i] = nullptr;
      m_sceneLights[i] = nullptr;
    }
  }
}

void
Lighting::captureLightsViewSpaceBasis(const CCamera& camera)
{
  // Update a copy of the camera to ensure m_U, m_V, m_N are current.
  CCamera cameraCopy = camera;
  cameraCopy.Update();
  for (size_t i = 0; i < m_NoLights; ++i) {
    m_capturedRelativeBasis[i] =
      cameraCopy.captureRelativeBasis(glm::mat3(m_Lights[i]->m_U, m_Lights[i]->m_V, m_Lights[i]->m_N));
  }
}

void
Lighting::restoreLightsViewSpaceBasis(const CCamera& camera, RenderSettings* rs)
{
  for (size_t i = 0; i < m_NoLights; ++i) {
    glm::mat3 basis = camera.reconstructBasis(m_capturedRelativeBasis[i]);
    m_sceneLights[i]->applyBasis(basis);
  }
  rs->m_DirtyFlags.SetFlag(LightsDirty);
}
