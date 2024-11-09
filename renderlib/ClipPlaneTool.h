#pragma once

#include "Manipulator.h"

struct ClipPlaneTool : ManipulationTool
{

  ClipPlaneTool()
    : ManipulationTool(0)
  {
  }

  virtual void action(SceneView& scene, Gesture& gesture) final;
  virtual void draw(SceneView& scene, Gesture& gesture) final;
};
