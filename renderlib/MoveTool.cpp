#include "MoveTool.h"

struct ManipColors
{
  static constexpr glm::vec3 xAxis = { 1.0f, 0.0f, 0.0f };
  static constexpr glm::vec3 yAxis = { 0.0f, 1.0f, 0.0f };
  static constexpr glm::vec3 zAxis = { 0.0f, 0.0f, 1.0f };
  static constexpr glm::vec3 bright = { 1.0f, 1.0f, 1.0f };
};

static const float s_lineThickness = 4.0f;

void
MoveTool::action(SceneView& scene, Gesture& gesture)
{
  // Most of the time the manipulator is called without an active code set
  bool validCode = isCodeValid(m_activeCode);
  if (!validCode)
    return;

  // The move tool is only active if some movable object is selected
  if (!scene.anythingActive())
    return;

  // Viewport resolution is needed to compute perspective projection of
  // the cursor position.
  const glm::ivec2 resolution = scene.viewport.region.size();

  // Move responds only to left pointer button click-drag.
  Gesture::Input::Button& button = gesture.input.mbs[Gesture::Input::kButtonLeft];
  if (button.action == Gesture::Input::Action::kPress) {
    origins.update(scene);
    m_translation = glm::vec3(0, 0, 0);
  }
  if (button.action == Gesture::Input::Action::kDrag) {
    // If we have not initialized objects position when the gesture began,
    // do so now. We need to know the initial object(s) position to
    // register undo later.
    // Note: because I didn't explain any part of the scene geometry data
    // structure, consider this as some pseudocode to show the intent.
    if (origins.empty()) {
      origins.update(scene);
    }

    // Need some position where the manipulator is drawn in space.
    // We assume this quantity did not change from the last time we
    // drew the manipulator.
    glm::vec3 targetPosition = origins.currentReference(scene).p;

    // Create an orthonormal frame of reference that is oriented towards
    // the camera position.
    // Compute the manipulator projective distance from the camera
    LinearSpace3f camFrame = scene.camera.getFrame();

    float v_len = scene.camera.getDistance(targetPosition);
    float aperture = scene.camera.getHalfHorizontalAperture();

    float dragScale = 0.5 * aperture * v_len / (resolution.x / 2);

    // Click in some proportional NDC: x [-1, 1] y [-aspect, aspect]
    glm::vec2 click0 = scene.viewport.toNDC(button.pressedPosition) * aperture;
    glm::vec2 click1 = scene.viewport.toNDC(button.pressedPosition + button.drag) * aperture;

    // Most of the math to get the manipulator to move will be about:
    // * line-line nearest point
    // * line-plane intersection
    // The equation of a line is expressed in parametric form as:
    //     P = l + l0 * t
    // The equation of a plane is expressed as a point and a normal.
    // Here we prepare some useful quantities:
    // p is the position of the manipulator where we may have a few planes
    // crossing, the normal of such planes may be the axis of the manipulator
    glm::vec3 p = targetPosition;

    // l is the camera position, the same "l" in the line parametric eq.
    glm::vec3 l = scene.camera.m_From;

    // l0 is the same "l0" as in the line parametric eq. as the direction of the
    // ray extending into the screen from the position of the initial click.
    // Let's call that "line 0"
    glm::vec3 l0 = normalize(xfmVector(camFrame, glm::vec3(click0.x, click0.y, -1.0)));

    // l1 is same concept as l1 but for the current position of the drag.
    // Let's call that "line 1"
    glm::vec3 l1 = normalize(xfmVector(camFrame, glm::vec3(click1.x, click1.y, -1.0)));

    LinearSpace3f ref; //< motion in reference space (world for now)
    if (m_localSpace) {
      ref = origins.currentReference(scene).l;
    }

    // Here we compute the effect of the cursor drag motion by projecting
    // the ray extending from the cursor position into the screen. We
    // project that along a line or onto a plane depending on what the user
    // clicks. We do it twice with and without the drag vector. The
    // difference is our motion in ref space.
    glm::vec3 motion(0);
    switch (m_activeCode) {
      case MoveTool::kMove:
        // Motion on the image plane is simpler than anything else here:
        // we simply scale the drag vector by the dragScale and transform
        // to the camera frame
        // TODO Why do we have to invert drag.y??
        motion = xfmVector(camFrame, glm::vec3(button.drag.x, -button.drag.y, 0)) * dragScale;
        break;

      case MoveTool::kMoveX:
        // To move along the X axis we need to compute two points along
        // the axis, the one that is closest to the line 0 and the one
        // for line 1.
        // The difference between those two points is the drag distance
        // along the axis. Neat right?
        motion = lineLineNearestPoint(p, ref.vx, l, l1) - lineLineNearestPoint(p, ref.vx, l, l0);
        break;

      case MoveTool::kMoveY:
        motion = lineLineNearestPoint(p, ref.vy, l, l1) - lineLineNearestPoint(p, ref.vy, l, l0);
        break;

      case MoveTool::kMoveZ:
        motion = lineLineNearestPoint(p, ref.vz, l, l1) - lineLineNearestPoint(p, ref.vz, l, l0);
        break;

      case MoveTool::kMoveYZ:
        // To move on two axis we compute the distance between two line-plane
        // intersections. The plane is that formed by the two axis.
        motion = linePlaneIsect(p, ref.vx, l, l1) - linePlaneIsect(p, ref.vx, l, l0);
        break;

      case MoveTool::kMoveXZ:
        motion = linePlaneIsect(p, ref.vy, l, l1) - linePlaneIsect(p, ref.vy, l, l0);
        break;

      case MoveTool::kMoveXY:
        motion = linePlaneIsect(p, ref.vz, l, l1) - linePlaneIsect(p, ref.vz, l, l0);
        break;
    }

    // The variable motion is a world space vector of how much we moved the
    // manipulator. Here we execute the action of applying motion to
    // whatever is that we are moving...
    // [...]
    // Note that the motion is the total motion from drag origin.
    // but we will update this per-frame!
    origins.translate(scene, motion);
    m_translation = motion;
  }
  if (button.action == Gesture::Input::Action::kRelease) {
    if (!origins.empty()) {
      // Make the edit final, for example by creating an undo action...
      // [...]
    }

    // Consume the gesture.
    gesture.input.reset(Gesture::Input::kButtonLeft);
    origins.clear();
    m_translation = glm::vec3(0, 0, 0);
  }
}

void
MoveTool::draw(SceneView& scene, Gesture& gesture)
{
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
  AffineSpace3f target = origins.currentReference(scene);
  // Append the current translation to the manipulator position.
  // This assumes that origins.currentReference is NOT translated
  target.p += m_translation;

  glm::vec3 viewDir = (scene.camera.m_From - target.p);
  LinearSpace3f camFrame = scene.camera.getFrame();

  // Draw the manipulator to be at some constant size on screen
  float scale = length(viewDir) * scene.camera.getHalfHorizontalAperture() * (m_size / resolution.x);

  AffineSpace3f axis;
  axis.p = target.p;
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
    bool drawAsActive = (m_activeCode == selectionCode) || forceActive;
    bool fullDraw = (drawAsActive || !isCodeValid(m_activeCode));
    if (drawAsActive) {
      color = glm::vec3(1, 1, 0);
    }

    uint32_t code = manipulatorCode(selectionCode, m_codesOffset);

    // Arrow line
    gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(axis.p + dir * (scale * 0.05f), color, opacity, code),
                                    Gesture::Graphics::VertsCode(axis.p + dir * scale, color, opacity, code) },
                                  s_lineThickness,
                                  false);

    // Circle at the base of the arrow
    float diskScale = scale * (fullDraw ? 0.06f : 0.03f);
    gesture.drawCircleAsStrip(
      axis.p + dir * scale, dirX * diskScale, dirY * diskScale, 12, color, 1, code, s_lineThickness);
    if (fullDraw) {
      // Arrow
      glm::vec3 ve = camFrame.vz - dir * dot(dir, camFrame.vz);
      glm::vec3 vd = normalize(cross(ve, dir)) * (scale * 0.06f);
      gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(axis.p + dir * scale + vd, color, opacity, code),
                                      Gesture::Graphics::VertsCode(axis.p + dir * (scale * 1.2f), color, opacity, code),
                                      Gesture::Graphics::VertsCode(axis.p + dir * scale - vd, color, opacity, code) },
                                    s_lineThickness,
                                    false);
    }

    // Extension to arrow line at the opposite end
    gesture.graphics.addLineStrip(
      { Gesture::Graphics::VertsCode(axis.p - dir * (scale * 0.05f), color, opacity, code),
        Gesture::Graphics::VertsCode(axis.p - dir * (scale * 0.25f), color, opacity, code) },
      s_lineThickness,
      false);

    gesture.drawCircleAsStrip(axis.p - dir * scale * 0.25f,
                              dirX * scale * 0.03f,
                              dirY * scale * 0.03f,
                              12,
                              color,
                              opacity,
                              code,
                              s_lineThickness);
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

  // The control to move a plane spanning between two axis
  auto drawDiag = [&](const glm::vec3& dir,
                      const glm::vec3& dirX,
                      const glm::vec3& dirY,
                      const float facingScale,
                      const uint32_t selectionCode,
                      glm::vec3 color,
                      float opacity) {
    if (facingScale < 0.1f)
      return;

    bool fullDraw = (m_activeCode == selectionCode || !isCodeValid(m_activeCode));

    glm::vec3 arrowsColor = glm::vec3(0.6);
    if (m_activeCode == selectionCode)
      arrowsColor = glm::vec3(1, 1, 0);

    uint32_t code = manipulatorCode(selectionCode, m_codesOffset);

    glm::vec3 p = axis.p + (dirX + dirY) * (scale * 0.7f);
    if (fullDraw) {
      glm::vec3 v0 = (dirX + dirY) * (scale * 0.06f * facingScale);
      glm::vec3 v1 = (dirX - dirY) * (scale * 0.06f * facingScale);
      float li = lerp(1.0, 0.5, facingScale);
      float lo = lerp(1.0, 0.7, facingScale);

      // Draw a bunch of 2d arrows
      float diskScale = scale * 0.17;
      gesture.graphics.addLineStrip(
        { Gesture::Graphics::VertsCode(p + dirX * diskScale * li, arrowsColor, opacity, code),
          Gesture::Graphics::VertsCode(p + dirX * diskScale * lo, arrowsColor, opacity, code) },
        s_lineThickness,
        false);

      gesture.graphics.addLineStrip(
        { Gesture::Graphics::VertsCode(p + dirX * diskScale - v1, arrowsColor, opacity, code),
          Gesture::Graphics::VertsCode(p + dirX * diskScale, arrowsColor, opacity, code),
          Gesture::Graphics::VertsCode(p + dirX * diskScale - v0, arrowsColor, opacity, code) },
        s_lineThickness,
        false);
      gesture.graphics.addLine(Gesture::Graphics::VertsCode(p + dirY * diskScale * li, arrowsColor, opacity, code),
                               Gesture::Graphics::VertsCode(p + dirY * diskScale * lo, arrowsColor, opacity, code));
      gesture.graphics.addLineStrip(
        { Gesture::Graphics::VertsCode(p + dirY * diskScale * li, arrowsColor, opacity, code),
          Gesture::Graphics::VertsCode(p + dirY * diskScale * lo, arrowsColor, opacity, code) },
        s_lineThickness,
        false);
      gesture.graphics.addLineStrip(
        { Gesture::Graphics::VertsCode(p + dirY * diskScale + v1, arrowsColor, opacity, code),
          Gesture::Graphics::VertsCode(p + dirY * diskScale, arrowsColor, opacity, code),
          Gesture::Graphics::VertsCode(p + dirY * diskScale - v0, arrowsColor, opacity, code) },
        s_lineThickness,
        false);
      gesture.graphics.addLineStrip(
        { Gesture::Graphics::VertsCode(p - dirX * diskScale * li, arrowsColor, opacity, code),
          Gesture::Graphics::VertsCode(p - dirX * diskScale * lo, arrowsColor, opacity, code) },
        s_lineThickness,
        false);
      gesture.graphics.addLineStrip(
        { Gesture::Graphics::VertsCode(p - dirX * diskScale + v1, arrowsColor, opacity, code),
          Gesture::Graphics::VertsCode(p - dirX * diskScale, arrowsColor, opacity, code),
          Gesture::Graphics::VertsCode(p - dirX * diskScale + v0, arrowsColor, opacity, code) },
        s_lineThickness,
        false);
      gesture.graphics.addLineStrip(
        { Gesture::Graphics::VertsCode(p - dirY * diskScale * li, arrowsColor, opacity, code),
          Gesture::Graphics::VertsCode(p - dirY * diskScale * lo, arrowsColor, opacity, code) },
        s_lineThickness,
        false);
      gesture.graphics.addLineStrip(
        { Gesture::Graphics::VertsCode(p - dirY * diskScale - v1, arrowsColor, opacity, code),
          Gesture::Graphics::VertsCode(p - dirY * diskScale, arrowsColor, opacity, code),
          Gesture::Graphics::VertsCode(p - dirY * diskScale + v0, arrowsColor, opacity, code) },
        s_lineThickness,
        false);
    }

    float diskScale = scale * 0.04 * facingScale;
    gesture.drawCircleAsStrip(p, dirX * diskScale, dirY * diskScale, 8, color, opacity, code, s_lineThickness);
  };

  bool forceActiveX =
    (m_activeCode == MoveTool::kMoveXZ) || (m_activeCode == MoveTool::kMoveXY) || (m_activeCode == MoveTool::kMove);
  bool forceActiveY =
    (m_activeCode == MoveTool::kMoveYZ) || (m_activeCode == MoveTool::kMoveXY) || (m_activeCode == MoveTool::kMove);
  bool forceActiveZ =
    (m_activeCode == MoveTool::kMoveYZ) || (m_activeCode == MoveTool::kMoveXZ) || (m_activeCode == MoveTool::kMove);

  gesture.graphics.addCommand(Gesture::Graphics::PrimitiveType::kTriangles);
  drawSolidArrow(axis.l.vx, axis.l.vy, axis.l.vz, MoveTool::kMoveX, forceActiveX, ManipColors::xAxis, 0.3f);
  drawSolidArrow(axis.l.vy, axis.l.vz, axis.l.vx, MoveTool::kMoveY, forceActiveY, ManipColors::yAxis, 0.3f);
  drawSolidArrow(axis.l.vz, axis.l.vx, axis.l.vy, MoveTool::kMoveZ, forceActiveZ, ManipColors::zAxis, 0.3f);

  gesture.graphics.addCommand(Gesture::Graphics::PrimitiveType::kLines);
  drawAxis(axis.l.vx, axis.l.vy, axis.l.vz, MoveTool::kMoveX, forceActiveX, ManipColors::xAxis, 1);
  drawAxis(axis.l.vy, axis.l.vz, axis.l.vx, MoveTool::kMoveY, forceActiveY, ManipColors::yAxis, 1);
  drawAxis(axis.l.vz, axis.l.vx, axis.l.vy, MoveTool::kMoveZ, forceActiveZ, ManipColors::zAxis, 1);

  // Draw planar move controls, only if facing angle makes them usable
  glm::vec3 vn = normalize(axis.p - scene.camera.m_From);

  float facingScale = glm::smoothstep(0.05f, 0.3f, (float)fabs(dot(vn, axis.l.vx)));
  drawDiag(axis.l.vx, axis.l.vy, axis.l.vz, facingScale, MoveTool::kMoveYZ, ManipColors::xAxis, 1);
  facingScale = glm::smoothstep(0.05f, 0.3f, (float)fabs(dot(vn, axis.l.vy)));
  drawDiag(axis.l.vy, axis.l.vz, axis.l.vx, facingScale, MoveTool::kMoveXZ, ManipColors::yAxis, 1);
  facingScale = glm::smoothstep(0.05f, 0.3f, (float)fabs(dot(vn, axis.l.vz)));
  drawDiag(axis.l.vz, axis.l.vx, axis.l.vy, facingScale, MoveTool::kMoveXY, ManipColors::zAxis, 1);

  // Draw the origin of the manipulator as a circle always facing the view
  {
    uint32_t code = manipulatorCode(MoveTool::kMove, m_codesOffset);
    glm::vec3 color = ManipColors::bright;
    if (m_activeCode == MoveTool::kMove) {
      color = glm::vec3(1, 1, 0);
    }

    float diskScale = scale * 0.09;
    gesture.drawCircleAsStrip(
      axis.p, camFrame.vx * diskScale, camFrame.vy * diskScale, 24, color, 1, code, s_lineThickness);
    diskScale = scale * 0.02;
    gesture.drawCircleAsStrip(
      axis.p, camFrame.vx * diskScale, camFrame.vy * diskScale, 24, color, 1, code, s_lineThickness);
  }
}
