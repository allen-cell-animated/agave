#pragma once

#include "Manipulator.h"
#include "Origins.h"

struct AxisHelperTool : public ManipulationTool
{
  AxisHelperTool(bool localspace = false, float size = ManipulationTool::s_manipulatorSize)
    : ManipulationTool(0)
    , m_localSpace(localspace)
  {
    setSize(size / 3.0);
  }

  virtual void action(SceneView& scene, Gesture& gesture) final;
  virtual void draw(SceneView& scene, Gesture& gesture) final;

  void setUseLocalSpace(bool localSpace) { m_localSpace = localSpace; }

  // Some data structure to store the initial state of the objects
  // to move.
  Origins origins;

  bool m_localSpace;
};
