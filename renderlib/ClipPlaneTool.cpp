#include "ClipPlaneTool.h"

void
ClipPlaneTool::action(SceneView& scene, Gesture& gesture)
{
}
void
ClipPlaneTool::draw(SceneView& scene, Gesture& gesture)
{
  if (!scene.scene) {
    return;
  }
  // TODO draw this as an oriented grid centered in the view (or on the volume?)
}
