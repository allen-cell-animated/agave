#pragma once

#include "Object3d.h"

#include "Light.h"
#include "AreaLightTool.h"

#include <vector>
#include <functional>

// MUST NOT OUTLIVE ITS LIGHT
class SceneLight : public SceneObject
{
public:
  SceneLight(Light* light)
    : m_light(light)
    , m_enabled(true)
  {
    // we want the rotate manipulator to be centered at the target of the light, by default
    m_transform.m_center = light->m_Target;
    m_tool = std::make_unique<AreaLightTool>(light);
  }

  void updateTransform();
  Light* m_light;
  bool m_enabled;
  std::vector<std::function<void(const Light&)>> m_observers;
  std::unique_ptr<AreaLightTool> m_tool;

  virtual ManipulationTool* getSelectedTool() { return m_tool.get(); }
};
