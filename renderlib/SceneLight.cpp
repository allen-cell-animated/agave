#include "SceneLight.h"

void
SceneLight::updateTransform()
{
  // update from transform position and rotation.

  // calculate the new direction for the light to point in
  glm::vec3 normdir = m_transform.m_rotation * glm::vec3(0, 0, 1);

  // compute the phi and theta for the new direction
  // because they will be used in future light updates
  float phi, theta;
  Light::cartesianToSpherical(normdir, phi, theta);
  m_light->m_Phi = phi;
  m_light->m_Theta = theta;

  // get m_P in world space:
  m_light->m_P = m_light->m_Distance * normdir + m_light->m_Target;

  m_light->updateBasisFrame();

  // this lets the GUI have a chance to update in an abstract way
  for (auto it = m_observers.begin(); it != m_observers.end(); ++it) {
    (*it)(*m_light);
  }

  m_transform.m_center = m_light->m_Target;
}
