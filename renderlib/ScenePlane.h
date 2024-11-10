#include "Object3d.h"

#include "MathUtil.h"

#include <vector>
#include <functional>

class ScenePlane : public SceneObject
{
public:
  ScenePlane() {}

  void updateTransform();
  std::vector<std::function<void(const Plane&)>> m_observers;

  Plane m_plane;
};
