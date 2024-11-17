#include "ScenePlane.h"

#include "Logging.h"

ScenePlane::ScenePlane(glm::vec3 pos)
{
  m_center = pos;
  m_transform.m_center = pos;
  m_plane = Plane().transform(m_transform.getMatrix());
  assert(m_plane.isInPlane(pos));
  m_enabled = true;
  m_tool = std::make_unique<ClipPlaneTool>(m_plane, pos);
}

void
ScenePlane::updateTransform()
{
  Plane p0;
  Plane p = p0.transform(m_transform.getMatrix());
  // m_center = m_center + m_transform.m_center;
  //  rotation only about current plane center... for now
  // p = Plane(p.normal, m_center + m_transform.m_center);

  //  this lets the GUI have a chance to update in an abstract way
  for (auto it = m_observers.begin(); it != m_observers.end(); ++it) {
    (*it)(p);
  }

  m_plane = p;
  m_tool->m_plane = p;
  m_tool->m_pos = m_transform.m_center;
  m_center = m_transform.m_center;
}
