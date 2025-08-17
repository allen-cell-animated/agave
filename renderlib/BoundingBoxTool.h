#pragma once

#include "Manipulator.h"

class Scene;

struct BoundingBoxTool : public ManipulationTool
{
  BoundingBoxTool()
    : ManipulationTool(0)
  {
  }

  virtual void action(SceneView& scene, Gesture& gesture) final;
  virtual void draw(SceneView& scene, Gesture& gesture) final;

private:
  void drawTickMarks(const class CBoundingBox& bbox,
                     Gesture& gesture,
                     const glm::vec3& color,
                     float opacity,
                     uint32_t code);
};
