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
  void drawEdgeTickMarks(const glm::vec3& vertex1,
                         const glm::vec3& vertex2,
                         const CBoundingBox& bbox,
                         float maxNumTickMarks,
                         Gesture& gesture,
                         const glm::vec3& color,
                         float opacity,
                         uint32_t code);
};
