#include "ScenePlane.h"

void
ScenePlane::updateTransform()
{
  Plane p = m_plane.transform(m_transform.getMatrix());
  // this lets the GUI have a chance to update in an abstract way
  for (auto it = m_observers.begin(); it != m_observers.end(); ++it) {
    (*it)(p);
  }

  m_plane = p;
}
