#pragma once

#include "Manipulator.h"

struct ClipPlaneTool : ManipulationTool
{

  ClipPlaneTool(Plane* plane)
    : ManipulationTool(0)
    , m_plane(plane)
  {
  }

  virtual void action(SceneView& scene, Gesture& gesture) final;
  virtual void draw(SceneView& scene, Gesture& gesture) final;

  Plane* m_plane;
};
