#pragma once

#include "Manipulator.h"

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
    , m_angle(0.0f)
  {
  }

  virtual void action(SceneView& scene, Gesture& gesture) final;
  virtual void draw(SceneView& scene, Gesture& gesture) final;

  // Some data structure to store the initial state of the objects
  // to move.
  Origins origins;

  // the current signed angle of rotation during drag
  float m_angle;

  // The current rotation of the objects to move.
  // We need to potentially access this across calls to action and draw
  glm::quat m_rotation;
};
