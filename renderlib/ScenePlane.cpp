#include "ScenePlane.h"

#include "Logging.h"

ScenePlane::ScenePlane(glm::vec3 pos)
{
  m_center = pos;
  m_transform.m_center = pos;
  m_plane = Plane(); //().transform(m_transform.getMatrix());
  assert(m_plane.isInPlane(pos));
  m_enabled = true;
  m_tool = std::make_unique<ClipPlaneTool>(m_plane, pos);
}

void
ScenePlane::updateTransform()
{
  Plane p;

  // strategy: keep all transform info in m_transform;
  // the plane stored here is just in default coords.

  // rotate the plane normal according to the transform's rotation.
  // then translate the plane center.
  // then rebuild the plane using the new plane center.

  m_center = m_transform.m_center;

  //  this lets the GUI have a chance to update in an abstract way
  for (auto it = m_observers.begin(); it != m_observers.end(); ++it) {
    // TODO if all info is in transform, then observers should be able to see it
    (*it)(p);
  }

  m_plane = p;
  // the clip plane tool should get the transformed plane and position to draw with
  m_tool->m_plane = p.transform(m_transform.getMatrix());
  m_tool->m_pos = m_transform.m_center;
}
