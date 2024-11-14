#pragma once

#include "Object3d.h"

#include "MathUtil.h"
#include "ClipPlaneTool.h"

#include <vector>
#include <functional>

class ScenePlane : public SceneObject
{
public:
  ScenePlane(glm::vec3 pos);

  void updateTransform();
  std::vector<std::function<void(const Plane&)>> m_observers;

  glm::vec3 m_center;
  Plane m_plane;
  bool m_enabled;
  std::unique_ptr<ClipPlaneTool> m_tool;

  virtual ManipulationTool* getSelectedTool() { return m_tool.get(); }
};
