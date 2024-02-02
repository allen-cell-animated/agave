#pragma once

#include "Manipulator.h"

struct ScaleBarTool : ManipulationTool
{

  ScaleBarTool()
    : ManipulationTool(0)
  {
  }

  virtual void action(SceneView& scene, Gesture& gesture) final;
  virtual void draw(SceneView& scene, Gesture& gesture) final;
};
