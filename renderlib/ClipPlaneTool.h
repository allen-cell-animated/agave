#pragma once

#include "Manipulator.h"

class SceneObject;

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

  void action(SceneView& scene, Gesture& gesture) final;
  void draw(SceneView& scene, Gesture& gesture) final;

  void setVisible(bool v) { m_visible = v; }

  Plane m_plane;
  glm::vec3 m_pos;

  // Back-pointer to the owning ScenePlane (set by ScenePlane ctor).
  // Used by draw() to check whether this tool's plane is selected.
  SceneObject* m_owner = nullptr;

  bool m_visible = true;
};