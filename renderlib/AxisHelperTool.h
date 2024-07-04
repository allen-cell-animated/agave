#pragma once

#include "Manipulator.h"
#include "Origins.h"

struct AxisHelperTool : public ManipulationTool
{
  AxisHelperTool(bool localspace = false, float size = ManipulationTool::s_manipulatorSize)
    : ManipulationTool(0)
  {
    setSize(size);
  }

  virtual void action(SceneView& scene, Gesture& gesture) final;
  virtual void draw(SceneView& scene, Gesture& gesture) final;
};
