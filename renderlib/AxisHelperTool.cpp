#include "AxisHelperTool.h"

struct ManipColors
{
  static constexpr glm::vec3 xAxis = { 1.0f, 0.0f, 0.0f };
  static constexpr glm::vec3 yAxis = { 0.0f, 1.0f, 0.0f };
  static constexpr glm::vec3 zAxis = { 0.0f, 0.0f, 1.0f };
  static constexpr glm::vec3 bright = { 1.0f, 1.0f, 1.0f };
};

void
AxisHelperTool::action(SceneView& scene, Gesture& gesture)
{
}

void
AxisHelperTool::draw(SceneView& scene, Gesture& gesture)
{
  if (!scene.scene) {
    return;
  }
  if (!scene.scene->m_showAxisHelper) {
    return;
  }

  const glm::vec2 resolution = scene.viewport.region.size();
  // Move tool is only active if something is selected
  if (!scene.anythingActive()) {
    return;
  }

  // Re-read targetPosition because object may have moved by the manipulator action,
  // or animation. Target 3x3 linear space is a orthonormal frame. Target
  // position is where the manipulator should be drawn in space
  if (origins.empty()) {
    origins.update(scene);
  }

  LinearSpace3f camFrame = scene.camera.getFrame();

  AffineSpace3f target; // = origins.currentReference(scene);
  target.p = scene.camera.m_Target;

  glm::vec3 viewDir = (scene.camera.m_From - target.p);

  // Draw the manipulator to be at some constant size on screen
  float scaleFactor = 0.333; // make it smaller
  float scale = scaleFactor * length(viewDir) * scene.camera.getHalfHorizontalAperture() * (m_size / resolution.x);

  AffineSpace3f axis;
  axis.p = target.p;
  // translate this in screen space toward the lower left corner!
  axis.p -= camFrame.vx * 0.4f * glm::length(viewDir) * resolution.x / resolution.y;
  axis.p -= camFrame.vy * 0.4f * glm::length(viewDir);

  if (m_localSpace) {
    axis.l = origins.currentReference(scene).l;
  }

  // Lambda to draw one axis of the manipulator, a wire-frame arrow.
  auto drawAxis = [&](const glm::vec3& dir,
                      const glm::vec3& dirX,
                      const glm::vec3& dirY,
                      const uint32_t selectionCode,
                      const bool forceActive,
                      glm::vec3 color,
                      float opacity) {
    bool drawAsActive = forceActive;
    bool fullDraw = (drawAsActive || !isCodeValid(m_activeCode));
    if (drawAsActive) {
      color = glm::vec3(1, 1, 0);
    }

    uint32_t code = manipulatorCode(selectionCode, m_codesOffset);

    // Arrow line
    gesture.graphics.addLine(Gesture::Graphics::VertsCode(axis.p + dir * (scale * 0.05f), color, opacity, code),
                             Gesture::Graphics::VertsCode(axis.p + dir * scale, color, opacity, code));

    // Circle at the base of the arrow
    // float diskScale = scale * (fullDraw ? 0.06f : 0.03f);
    // gesture.drawCircle(axis.p + dir * scale, dirX * diskScale, dirY * diskScale, 12, color, 1, code);
    // if (fullDraw) {
    //   // Arrow
    //   glm::vec3 ve = camFrame.vz - dir * dot(dir, camFrame.vz);
    //   glm::vec3 vd = normalize(cross(ve, dir)) * (scale * 0.06f);
    //   gesture.graphics.addLine(Gesture::Graphics::VertsCode(axis.p + dir * scale + vd, color, opacity, code),
    //                            Gesture::Graphics::VertsCode(axis.p + dir * (scale * 1.2f), color, opacity, code));
    //   gesture.graphics.extLine(Gesture::Graphics::VertsCode(axis.p + dir * scale - vd, color, opacity, code));
    // }

    // Extension to arrow line at the opposite end
    // gesture.graphics.addLine(Gesture::Graphics::VertsCode(axis.p - dir * (scale * 0.05f), color, opacity, code),
    //                          Gesture::Graphics::VertsCode(axis.p - dir * (scale * 0.25f), color, opacity, code));

    // gesture.drawCircle(
    //   axis.p - dir * scale * 0.25f, dirX * scale * 0.03f, dirY * scale * 0.03f, 12, color, opacity, code);
  };

  // Complete the axis with a transparent surface for the tip of the arrow
  auto drawSolidArrow = [&](const glm::vec3& dir,
                            const glm::vec3& dirX,
                            const glm::vec3& dirY,
                            const uint32_t selectionCode,
                            const bool forceActive,
                            glm::vec3 color,
                            float opacity) {
    bool drawAsActive = (m_activeCode == selectionCode) || forceActive;
    bool fullDraw = (drawAsActive || !isCodeValid(m_activeCode));
    if (drawAsActive) {
      color = glm::vec3(1, 1, 0);
    }

    uint32_t code = manipulatorCode(selectionCode, m_codesOffset);

    if (fullDraw) {
      float diskScale = scale * 0.06;

      // The cone
      gesture.drawCone(
        axis.p + dir * scale, dirX * diskScale, dirY * diskScale, dir * (scale * 0.2f), 12, color, opacity, code);

      // The base of the cone (as a flat cone)
      gesture.drawCone(
        axis.p + dir * scale, dirX * diskScale, dirY * diskScale, glm::vec3(0, 0, 0), 12, color, opacity, code);
    }
  };

  bool forceActiveX = false;
  bool forceActiveY = false;
  bool forceActiveZ = false;

  uint32_t code = Gesture::Graphics::SelectionBuffer::k_noSelectionCode;

  gesture.graphics.addCommand(GL_TRIANGLES);
  drawSolidArrow(axis.l.vx, axis.l.vy, axis.l.vz, code, forceActiveX, ManipColors::xAxis, 0.3f);
  drawSolidArrow(axis.l.vy, axis.l.vz, axis.l.vx, code, forceActiveY, ManipColors::yAxis, 0.3f);
  drawSolidArrow(axis.l.vz, axis.l.vx, axis.l.vy, code, forceActiveZ, ManipColors::zAxis, 0.3f);

  gesture.graphics.addCommand(GL_LINES);
  drawAxis(axis.l.vx, axis.l.vy, axis.l.vz, code, forceActiveX, ManipColors::xAxis, 1);
  drawAxis(axis.l.vy, axis.l.vz, axis.l.vx, code, forceActiveY, ManipColors::yAxis, 1);
  drawAxis(axis.l.vz, axis.l.vx, axis.l.vy, code, forceActiveZ, ManipColors::zAxis, 1);

  // Draw planar move controls, only if facing angle makes them usable
  glm::vec3 vn = normalize(axis.p - scene.camera.m_From);

  // Draw the origin of the manipulator as a circle always facing the view
  // {
  //   glm::vec3 color = ManipColors::bright;
  //   float diskScale = scale * 0.09;
  //   gesture.drawCircle(axis.p, camFrame.vx * diskScale, camFrame.vy * diskScale, 24, color, 1, code);
  //   diskScale = scale * 0.02;
  //   gesture.drawCircle(axis.p, camFrame.vx * diskScale, camFrame.vy * diskScale, 24, color, 1, code);
  // }
}
