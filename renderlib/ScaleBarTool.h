#pragma once

#include "Manipulator.h"

struct ScaleBarTool : ManipulationTool
{

  ScaleBarTool()
    : ManipulationTool(0)
  {
  }

  void action(SceneView& scene, Gesture& gesture) final;
  void draw(SceneView& scene, Gesture& gesture) final;
};
