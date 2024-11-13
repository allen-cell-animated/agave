#pragma once

#include "Manipulator.h"

struct ClipPlaneTool : ManipulationTool
{

  ClipPlaneTool(Plane* plane, glm::vec3& pos)
    : ManipulationTool(0)
    , m_plane(plane)
    , m_pos(pos)
  {
  }

  virtual void action(SceneView& scene, Gesture& gesture) final;
  virtual void draw(SceneView& scene, Gesture& gesture) final;

  Plane* m_plane;
  glm::vec3 m_pos;
};
