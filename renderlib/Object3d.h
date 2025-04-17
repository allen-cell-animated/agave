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

  virtual ManipulationTool* getSelectedTool() { return nullptr; }

  virtual void onSelection(bool selected) {}
};
