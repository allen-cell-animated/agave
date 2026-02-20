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
  glm::vec3 p = l.m_Target;

  glm::vec3 viewDir = (scene.camera.m_From - p);
  LinearSpace3f camFrame = scene.camera.getFrame();
  // remember the camFrame vectors are the world-space vectors that correspond to camera x, y, z directions

  glm::vec3 color = glm::vec3(1, 1, 1);
  float opacity = 1.0f;
  uint32_t code = Gesture::Graphics::k_noSelectionCode;
  gesture.graphics.addCommand(Gesture::Graphics::PrimitiveType::kLines);

  // Draw the circle so it inscribes the viewport (touching the smaller dimension edges)
  float dist = length(viewDir);
  float projectedRadius = 0.0f;
  float sphereRadius = 0.0f;
  if (scene.camera.m_Projection == ORTHOGRAPHIC) {
    glm::ivec2 viewportSize = scene.viewport.region.size();
    float aspect = static_cast<float>(viewportSize.x) / static_cast<float>(viewportSize.y);
    float halfWidth = scene.camera.m_OrthoScale * aspect;
    float halfHeight = scene.camera.m_OrthoScale;
    projectedRadius = glm::min(halfWidth, halfHeight);
    sphereRadius = projectedRadius;
  } else {
    float halfWidth = dist * scene.camera.getHalfHorizontalAperture();
    float halfHeight = dist * tan(scene.camera.GetVerticalFOV_radians() * 0.5f);
    projectedRadius = glm::min(halfWidth, halfHeight);
    float projectedRadiusSq = projectedRadius * projectedRadius;
    float distSq = glm::max(dist * dist, 1e-6f);
    sphereRadius = projectedRadius / sqrtf(1.0f + (projectedRadiusSq / distSq));
  }

  gesture.drawCircle(p, camFrame.vx * projectedRadius, camFrame.vy * projectedRadius, 128, color, opacity, code);

  opacity = 0.3f;

  glm::vec3 colorTop = l.m_ColorTop * l.m_ColorTopIntensity;
  glm::vec3 colorMid = l.m_ColorMiddle * l.m_ColorMiddleIntensity;
  glm::vec3 colorBottom = l.m_ColorBottom * l.m_ColorBottomIntensity;
  auto ringColorFromY = [&](float y) {
    if (y >= 0.0f) {
      return glm::mix(colorMid, colorTop, glm::clamp(y, 0.0f, 1.0f));
    }
    return glm::mix(colorMid, colorBottom, glm::clamp(-y, 0.0f, 1.0f));
  };

  const int latBands = 12;
  for (int i = 1; i < latBands; i++) {
    float t = static_cast<float>(i) / static_cast<float>(latBands);
    float lat = t * PI_F - HALF_PI_F;
    float ringRadius = sphereRadius * cosf(lat);
    float y = sinf(lat);
    glm::vec3 ringCenter = p + l.m_V * (sphereRadius * y);
    glm::vec3 ringColor = ringColorFromY(y);
    gesture.drawCircle(ringCenter, l.m_U * ringRadius, l.m_N * ringRadius, 128, ringColor, opacity, code);
  }

  const int lonBands = 12;
  for (int i = 0; i < lonBands; i++) {
    float lon = (static_cast<float>(i) / static_cast<float>(lonBands)) * PI_F;
    glm::vec3 axis = cosf(lon) * l.m_N + sinf(lon) * l.m_U;
    gesture.drawCircle(p, axis * sphereRadius, l.m_V * sphereRadius, 128, color, opacity, code);
  }
}
