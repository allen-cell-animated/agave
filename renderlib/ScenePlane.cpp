#include "ScenePlane.h"

#include "Logging.h"

ScenePlane::ScenePlane(glm::vec3 pos)
{
  m_enabled = true;
  m_center = pos;
  m_tool = std::make_unique<ClipPlaneTool>(&m_plane, pos);
  LOG_DEBUG << "pos = " << pos.x << ", " << pos.y << ", " << pos.z;
}

void
ScenePlane::updateTransform()
{
  // rotation only... for now
  m_transform.m_center = m_center;

  Plane p0;
  Plane p = p0.transform(m_transform.getMatrix());
  // this lets the GUI have a chance to update in an abstract way
  for (auto it = m_observers.begin(); it != m_observers.end(); ++it) {
    (*it)(p);
  }

  m_plane = p;
}
