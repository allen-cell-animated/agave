#include "SceneView.h"

// TODO clamp to region bounds
// flip Y to match window coordinates
glm::ivec2
SceneView::Viewport::toRaster(const glm::vec2& p) const
{
  return glm::ivec2((int)p.x, region.size().y - (int)p.y);
}

// scale to the lower left (-1, -1) to upper right (1,1) range
glm::vec2
SceneView::Viewport::toNDC(const glm::ivec2& p) const
{
  return glm::vec2((2.0f * p.x) / region.size().x - 1.0f, (2.0f * p.y) / region.size().y - 1.0f);
}
