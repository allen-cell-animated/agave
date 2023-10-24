#pragma once

#include "Object3d.h"
#include "SceneView.h"

#include <vector>

// currently this code is specific to area light but should be tuned to find
// selected objects in scene and use an average of their centers.
//
// TODO create a generic SceneObject class that supports rotation/translation
// and Origins holds a collection/group of them.  We need to consider whether we
// are truly rotating the object or in the case of the light source doing some
// custom version of a rotation (keeping the area light looking toward a fixed target).
//
// Allow temporary rotation/translation and then
// the idea of committing the transform when the user releases the mouse.
struct Origins
{
  enum OriginFlags
  {
    kDefault = 0,
    kNormalize = 1
  };

  void clear() { m_origins.clear(); }
  bool empty() const { return m_origins.size() == 0; }
  void update(SceneView& scene);
  AffineSpace3f currentReference(SceneView& scene, OriginFlags flags = OriginFlags::kDefault)
  {
    return m_origins[0].getAffineSpace();
  }

  void translate(SceneView& scene, glm::vec3 motion);

  void rotate(SceneView& scene, glm::quat rotation);

  std::vector<Transform3d> m_origins;
};
