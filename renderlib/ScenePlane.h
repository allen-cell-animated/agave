#pragma once

#include "Object3d.h"
#include "ClipPlaneTool.h"

#include "MathUtil.h"

#include <vector>
#include <functional>

class ScenePlane : public SceneObject
{
public:
  ScenePlane(glm::vec3 pos);

  void updateTransform();
  std::vector<std::function<void(const Plane&)>> m_observers;

  glm::vec3 m_center;
  // transformed into world space:
  Plane m_plane;
  bool m_enabled;

  std::unique_ptr<ClipPlaneTool> m_tool;

  virtual ManipulationTool* getTool();

  void resetTo(const glm::vec3& c);

  // should the tool be showing?
  void setVisible(bool v);
};
