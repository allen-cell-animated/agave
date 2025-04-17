#pragma once

#include "Manipulator.h"

struct ClipPlaneTool : ManipulationTool
{

  ClipPlaneTool(Plane plane, glm::vec3& pos)
    : ManipulationTool(0)
    , m_plane(plane)
    , m_pos(pos)
  {
    // assumes pos is in plane! if not, there will be trouble.
    // assert(glm::dot(m_plane.normal, m_pos) == m_plane.d);
  }

  virtual void action(SceneView& scene, Gesture& gesture) final;
  virtual void draw(SceneView& scene, Gesture& gesture) final;

  void setVisible(bool v) { m_visible = v; }

  Plane m_plane;
  glm::vec3 m_pos;

  bool m_visible = true;
};