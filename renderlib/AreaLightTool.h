#pragma once

#include "Manipulator.h"

class Light;

struct AreaLightTool : public ManipulationTool
{

  AreaLightTool(Light* light)
    : ManipulationTool(0)
    , m_light(light)
  {
  }

  void action(SceneView& scene, Gesture& gesture) final;
  void draw(SceneView& scene, Gesture& gesture) final;

  Light* m_light;
};
