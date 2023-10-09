#pragma once

#include "Manipulator.h"
#include "Origins.h"

struct MoveTool : ManipulationTool
{
  // Selection codes, are used to identify which manipulator is under the cursor.
  // The values in this enum are important, lower values means higher picking priority.
  enum MoveCodes
  {
    kMove = 0,
    kMoveX = 1,
    kMoveY = 2,
    kMoveZ = 3,
    kMoveYZ = 4,
    kMoveXZ = 5,
    kMoveXY = 6,
    kLast = 7
  };

  MoveTool()
    : ManipulationTool(kLast)
    , m_translation(0)
    , m_localSpace(false)
  {
  }

  virtual void action(SceneView& scene, Gesture& gesture) final;
  virtual void draw(SceneView& scene, Gesture& gesture) final;

  void setLocalSpace(bool isLocalSpace) { m_localSpace = isLocalSpace; }

  // Some data structure to store the initial state of the objects
  // to move.
  Origins origins;

  // The current translation of the objects to move.
  // We need to access this across calls to action and draw
  glm::vec3 m_translation;

  bool m_localSpace;
};
