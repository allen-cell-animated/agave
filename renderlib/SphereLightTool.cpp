#include "SphereLightTool.h"

#include "Light.h"

void
SphereLightTool::action(SceneView& scene, Gesture& gesture)
{
}
void
SphereLightTool::draw(SceneView& scene, Gesture& gesture)
{
  if (!scene.scene) {
    return;
  }

  const Light& l = *m_light;
  glm::vec3 p = l.m_P;
  glm::vec3 t = l.m_Target;
  float scale = l.m_Width * 0.5;
  // compute 4 vertices of square area light pointing at 0, 0, 0
  glm::vec3 v0 = l.m_U * (-scale) + l.m_V * (-scale);
  glm::vec3 v1 = l.m_U * scale + l.m_V * (-scale);
  glm::vec3 v2 = l.m_U * scale + l.m_V * scale;
  glm::vec3 v3 = l.m_U * (-scale) + l.m_V * scale;

  glm::vec3 viewDir = (scene.camera.m_From - p);
  LinearSpace3f camFrame = scene.camera.getFrame();
  // remember the camFrame vectors are the world-space vectors that correspond to camera x, y, z directions

  glm::vec3 color = glm::vec3(1, 1, 1);
  float opacity = 1.0f;
  uint32_t code = Gesture::Graphics::k_noSelectionCode;
  gesture.graphics.addCommand(Gesture::Graphics::PrimitiveType::kLines);

  // Draw the circle so it inscribes the viewport (touching the smaller dimension edges)
  float dist = length(viewDir);
  float halfWidth = dist * scene.camera.getHalfHorizontalAperture();
  float halfHeight = dist * tan(scene.camera.GetVerticalFOV_radians() * 0.5f);
  float manipscale = glm::min(halfWidth, halfHeight);

  gesture.drawCircle(p, camFrame.vx * manipscale, camFrame.vy * manipscale, 128, color, opacity, code);
  gesture.drawCircle(p, camFrame.vx * manipscale * 0.75f, camFrame.vy * manipscale, 128, color, opacity, code);
  gesture.drawCircle(p, camFrame.vx * manipscale, camFrame.vy * manipscale * 0.75f, 128, color, opacity, code);
}
