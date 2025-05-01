#pragma once

#include "MathUtil.h"

class ManipulationTool;

class SceneObject
{
public:
  virtual ~SceneObject() {}

  // by default do nothing and assume object will read from m_transform...?
  virtual void updateTransform() {}

  Transform3d m_transform;

  // to be drawn when object is selected; must not outlive SceneObject
  virtual ManipulationTool* getSelectedTool() { return nullptr; }

  // to be drawn when object exists in the scene; must not outlive SceneObject
  virtual ManipulationTool* getTool() { return nullptr; }

  virtual void onSelection(bool selected) {}
};
