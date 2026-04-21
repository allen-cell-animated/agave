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

  using GFX = Gesture::Graphics;
  using Seq = GFX::CommandSequence;

  const Light& l = *m_light;
  glm::vec3 p = l.m_Target;

  glm::vec3 viewDir = (scene.camera.m_From - p);
  LinearSpace3f camFrame = scene.camera.getFrame();

  // Clip plane through sphere center, perpendicular to view direction.
  // Clip plane through sphere center, perpendicular to view direction.
  // drawCircle/drawCircleAsStrip keep geometry where dot(clipPlane, point) > 0.
  // frontPlane positive side faces camera; backPlane positive side faces away.
  glm::vec3 viewDirN = glm::normalize(viewDir);
  glm::vec4 frontPlane(viewDirN, -glm::dot(viewDirN, p));
  glm::vec4 backPlane = -frontPlane;

  glm::vec3 color = glm::vec3(1, 1, 1);
  float opacity = 1.0f;
  uint32_t code = GFX::k_noSelectionCode;

  // Draw the circle so it inscribes the viewport (touching the smaller dimension edges)
  float dist = glm::length(viewDir);
  float projectedRadius = 0.0f;
  float sphereRadius = 0.0f;
  glm::ivec2 viewportSize = scene.viewport.region.size();
  if (scene.camera.m_Projection == ORTHOGRAPHIC) {
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

  // Silhouette circle — always foreground (view-aligned, always in front)
  gesture.graphics.addCommand(GFX::PrimitiveType::kLines, Seq::k3dStacked);
  gesture.drawCircle(p, camFrame.vx * projectedRadius, camFrame.vy * projectedRadius, 128, color, opacity, code);

  opacity = 0.3f;

  glm::vec3 colorTop = l.m_ColorTop * l.m_ColorTopIntensity;
  glm::vec3 colorMid = l.m_ColorMiddle * l.m_ColorMiddleIntensity;
  glm::vec3 colorBottom = l.m_ColorBottom * l.m_ColorBottomIntensity;

  // Thin white latitude lines — front half then back half
  const int latBands = 8;
  for (int i = 1; i < latBands; i++) {
    float t = static_cast<float>(i) / static_cast<float>(latBands);
    float lat = t * PI_F - HALF_PI_F;
    float ringRadius = sphereRadius * cosf(lat);
    float y = sinf(lat);
    glm::vec3 ringCenter = p + l.m_V * (sphereRadius * y);
    gesture.graphics.addCommand(GFX::PrimitiveType::kLines, Seq::k3dStacked);
    gesture.drawCircle(ringCenter, l.m_U * ringRadius, l.m_N * ringRadius, 128, color, opacity, code, &frontPlane);
    gesture.graphics.addCommand(GFX::PrimitiveType::kLines, Seq::k3dStackedUnderlay);
    gesture.drawCircle(ringCenter, l.m_U * ringRadius, l.m_N * ringRadius, 128, color, opacity, code, &backPlane);
  }

  // Longitude lines (thin white) — front half then back half
  const int lonBands = 8;
  for (int i = 0; i < lonBands; i++) {
    float lon = (static_cast<float>(i) / static_cast<float>(lonBands)) * PI_F;
    glm::vec3 axis = cosf(lon) * l.m_N + sinf(lon) * l.m_U;
    gesture.graphics.addCommand(GFX::PrimitiveType::kLines, Seq::k3dStacked);
    gesture.drawCircle(p, axis * sphereRadius, l.m_V * sphereRadius, 128, color, opacity, code, &frontPlane);
    gesture.graphics.addCommand(GFX::PrimitiveType::kLines, Seq::k3dStackedUnderlay);
    gesture.drawCircle(p, axis * sphereRadius, l.m_V * sphereRadius, 128, color, opacity, code, &backPlane);
  }

  // Thicker colored equator band — front half then back half
  gesture.drawCircleAsStrip(
    p, l.m_U * sphereRadius, l.m_N * sphereRadius, 128, colorMid, 0.7f, code, 8.0f, &frontPlane, Seq::k3dStacked);
  gesture.drawCircleAsStrip(p,
                            l.m_U * sphereRadius,
                            l.m_N * sphereRadius,
                            128,
                            colorMid,
                            0.7f,
                            code,
                            8.0f,
                            &backPlane,
                            Seq::k3dStackedUnderlay);

  // Pole caps using drawCone — route to foreground or underlay based on visibility
  float capAngle = 0.2f; // radians from pole (~11 degrees)
  float capRingRadius = sphereRadius * sinf(capAngle);
  float capHeight = sphereRadius * cosf(capAngle);
  const int capSegments = 32;

  // North pole cap
  {
    glm::vec3 polePoint = p + l.m_V * sphereRadius;
    glm::vec3 ringCenter = p + l.m_V * capHeight;
    bool poleFacesCamera = glm::dot(viewDirN, polePoint - p) > 0;
    Seq capSeq = poleFacesCamera ? Seq::k3dStacked : Seq::k3dStackedUnderlay;
    gesture.graphics.addCommand(GFX::Command(GFX::PrimitiveType::kTriangles, 1.0f, true), capSeq);
    gesture.drawCone(ringCenter,
                     l.m_U * capRingRadius,
                     l.m_N * capRingRadius,
                     l.m_V * (sphereRadius - capHeight),
                     capSegments,
                     colorTop,
                     0.7f,
                     code);
  }

  // South pole cap
  {
    glm::vec3 polePoint = p - l.m_V * sphereRadius;
    glm::vec3 ringCenter = p - l.m_V * capHeight;
    bool poleFacesCamera = glm::dot(viewDirN, polePoint - p) > 0;
    Seq capSeq = poleFacesCamera ? Seq::k3dStacked : Seq::k3dStackedUnderlay;
    gesture.graphics.addCommand(GFX::Command(GFX::PrimitiveType::kTriangles, 1.0f, true), capSeq);
    gesture.drawCone(ringCenter,
                     l.m_U * capRingRadius,
                     l.m_N * capRingRadius,
                     -l.m_V * (sphereRadius - capHeight),
                     capSegments,
                     colorBottom,
                     0.7f,
                     code);
  }
}
