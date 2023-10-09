#pragma once

#include "Manipulator.h"
#include "Origins.h"

struct RotateTool : ManipulationTool
{
  // Selection codes, are used to identify which manipulator is under the cursor.
  // The values in this enum are important, lower values means higher picking priority.
  enum RotateCodes
  {
    kRotateX = 0,    // constrained to rotate about x axis
    kRotateY = 1,    // constrained to rotate about y axis
    kRotateZ = 2,    // constrained to rotate about z axis
    kRotateView = 3, // constrained to rotate about view direction
    kRotate = 4,     // general tumble rotation
    kLast = 5
  };

  RotateTool()
    : ManipulationTool(kLast)
    , m_rotation(glm::vec3(0, 0, 0))
    , m_localSpace(false)
  {
  }

  virtual void action(SceneView& scene, Gesture& gesture) final;
  virtual void draw(SceneView& scene, Gesture& gesture) final;

  void setUseLocalSpace(bool localSpace) { m_localSpace = localSpace; }

  // Some data structure to store the initial state of the objects
  // to move.
  Origins origins;

  // The current rotation of the objects to move.
  // We need to potentially access this across calls to action and draw
  glm::quat m_rotation;

  bool m_localSpace;
};
