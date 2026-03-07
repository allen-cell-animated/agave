#include "SceneLight.h"

void
SceneLight::updateTransform()
{
  // update from transform position and rotation.

  // calculate the new direction for the light to point in
  glm::mat3 rot = glm::mat3_cast(m_transform.m_rotation);
  glm::vec3 normdir = rot * glm::vec3(0.0f, 0.0f, 1.0f);

  // compute the phi and theta for the new direction
  // because they will be used in future light updates
  float phi, theta;
  Light::cartesianToSpherical(normdir, phi, theta);
  m_light->m_Phi = phi;
  m_light->m_Theta = theta;

  if (m_light->m_T == LightType_Area) {
    // For area lights, move the light position while keeping the target fixed.
    m_light->m_P = m_light->m_Distance * normdir + m_light->m_Target;
  } else {
    // For sphere/sky lights, update the light direction while keeping the target fixed.
    m_light->m_P = normdir + m_light->m_Target;
  }

  // Keep basis aligned with the transform so roll changes affect the skylight/sphere visualization.
  m_light->m_U = rot * glm::vec3(1.0f, 0.0f, 0.0f);
  m_light->m_V = rot * glm::vec3(0.0f, 1.0f, 0.0f);
  m_light->updateBasisFrame();

  // this lets the GUI have a chance to update in an abstract way
  for (auto it = m_observers.begin(); it != m_observers.end(); ++it) {
    (*it)(*m_light);
  }
}
