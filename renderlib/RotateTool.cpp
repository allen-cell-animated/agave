#include "RotateTool.h"

struct ManipColors
{
  static constexpr glm::vec3 xAxis = { 1.0f, 0.0f, 0.0f };
  static constexpr glm::vec3 yAxis = { 0.0f, 1.0f, 0.0f };
  static constexpr glm::vec3 zAxis = { 0.0f, 0.0f, 1.0f };
  static constexpr glm::vec3 bright = { 1.0f, 1.0f, 1.0f };
};

static const float s_orthogonalThreshold = cos(glm::radians(89.0f));

static float
getSignedAngle(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& vN)
{
  // get signed angle between the two vectors using (1,0,0) as plane normal
  // atan2 ( (v0 X v1) dot (vN), v0 dot v1 )

  float dp = dot(v0, v1);
  float angle = atan2(dot(cross(v0, v1), vN), dp);
  return angle;
}

static float
getDraggedAngle(const glm::vec3& vN, const glm::vec3& p, const glm::vec3& l, const glm::vec3& l0, const glm::vec3& l1)
{
  glm::vec3 globalAxis = normalize(p - l);
  bool axisIsOrthogonal = abs(dot(globalAxis, vN)) < s_orthogonalThreshold;

  // axis is represented by vN, and a point in plane p (the center of our circle)
  // line l-l0 is the ray of the initial mouse click
  // line l-l1 is the ray of the current mouse position
  // project our drag pts into plane of circle, then find the angle between
  // the two vectors
  glm::vec3 p0 = linePlaneIsect(p, vN, l, l0);
  glm::vec3 p1 = linePlaneIsect(p, vN, l, l1);
  // if we can't intersect the planes properly, then we must be on-axis (plane perpendicular to view plane)
  // and we must calculate angle another way (TODO)
  if (p0 == p1 || axisIsOrthogonal) {
    // we want a linear measure of the amount of drag along the line of the ring
    glm::vec3 projectionAxis = cross(vN, globalAxis);
    glm::vec3 delta = l1 - l0;
    float projection = dot(delta, projectionAxis);
    return projection * glm::two_pi<float>();
  }

  // get angle between (p,p0) and (p,p1)
  glm::vec3 v0 = normalize(p - p0);
  glm::vec3 v1 = normalize(p - p1);
  float angle = getSignedAngle(v0, v1, vN);
  return angle;
}

void
RotateTool::action(SceneView& scene, Gesture& gesture)
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
    m_rotation = glm::angleAxis(0.0f, glm::vec3(0, 0, 1));
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

    // ultimately there are two modes.
    // One is distance from ring (linear mouse drag) - NOT IMPLEMENTED YET
    // The second (default) is rotation around the ring

    // Find the point closest to the active ring.
    // if tumbling, then we don't need it.

    // world space x,y,z
    LinearSpace3f rotationFrame;
    // if local space, then rotationFrame should use object's current rotation!

    float angle = 0.0f;
    glm::quat motion = glm::angleAxis(0.0f, glm::vec3(0, 0, 1));
    switch (m_activeCode) {
      case RotateTool::kRotateX: // constrained to rotate about world x axis
      {
        glm::vec3 vN = rotationFrame.vx; // glm::vec3(1, 0, 0);
        angle = getDraggedAngle(rotationFrame.vx, p, l, l0, l1);
        motion = glm::angleAxis(angle, vN);

      } break;
      case RotateTool::kRotateY: // constrained to rotate about world y axis
      {
        glm::vec3 vN = rotationFrame.vy; // glm::vec3(0, 1, 0);
        angle = getDraggedAngle(rotationFrame.vy, p, l, l0, l1);
        motion = glm::angleAxis(angle, vN);

      } break;
      case RotateTool::kRotateZ: // constrained to rotate about world z axis
      {
        glm::vec3 vN = rotationFrame.vz; // glm::vec3(0, 0, 1);
        angle = getDraggedAngle(rotationFrame.vz, p, l, l0, l1);
        motion = glm::angleAxis(angle, vN);

      } break;
      case RotateTool::kRotateView: // constrained to rotate about view direction
      {
        // find angle between (p,l0) and (p,l1)
        // using as N, the view direction
        glm::vec3 vN = normalize(p - l);
        glm::vec3 v0 = normalize(p - l0);
        glm::vec3 v1 = normalize(p - l1);
        angle = getSignedAngle(v0, v1, vN);
        motion = glm::angleAxis(angle, vN);
      } break;
      case RotateTool::kRotate: // general tumble rotation
        // use camera trackball algorithm
        // scale pixels to radians of rotation (TODO)
        float xRadians = -button.drag.x * dragScale * glm::two_pi<float>();
        float yRadians = -button.drag.y * dragScale * glm::two_pi<float>();
        angle = sqrtf(yRadians * yRadians + xRadians * xRadians);
        glm::vec3 objectUpDirection = camFrame.vy * yRadians;
        glm::vec3 objectSidewaysDirection = camFrame.vx * xRadians;

        glm::vec3 moveDirection = objectUpDirection + objectSidewaysDirection;
        glm::vec3 eye = l - p;
        glm::vec3 axis = glm::normalize(glm::cross(moveDirection, eye));

        motion = glm::angleAxis(angle, axis);

        break;
    }

    // The variable motion is a world space vector of how much we moved the
    // manipulator. Here we execute the action of applying motion to
    // whatever is that we are moving...
    // [...]
    // Note that the motion is the total motion from drag origin.
    // but we will update this per-frame!
    origins.rotate(scene, motion);
    m_rotation = motion;
  }
  if (button.action == Gesture::Input::Action::kRelease) {
    if (!origins.empty()) {
      // Make the edit final, for example by creating an undo action...
      // [...]
    }

    // Consume the gesture.
    gesture.input.reset(Gesture::Input::kButtonLeft);
    origins.clear();
    m_rotation = glm::angleAxis(0.0f, glm::vec3(0, 0, 1));
  }
}

void
RotateTool::draw(SceneView& scene, Gesture& gesture)
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
  AffineSpace3f target = origins.currentReference(scene, Origins::kNormalize);

  glm::vec3 viewDir = (scene.camera.m_From - target.p);
  LinearSpace3f camFrame = scene.camera.getFrame();

  // Draw the manipulator to be at some constant size on screen
  float scale = length(viewDir) * scene.camera.getHalfHorizontalAperture() * (s_manipulatorSize / resolution.x);

  AffineSpace3f axis;
  axis.p = target.p;

  gesture.graphics.addCommand(GL_TRIANGLES);
  // draw a flat camera facing disk for freeform tumble rotation
  float tumblescale = scale * 0.84f;
  {
    float opacity = 0.05f;
    uint32_t code = manipulatorCode(RotateTool::kRotate, m_codesOffset);
    glm::vec3 color = ManipColors::bright;
    if (m_activeCode == RotateTool::kRotate) {
      color = glm::vec3(1, 1, 0);
      opacity = 0.2f;
    }
    gesture.drawCone(axis.p,
                     camFrame.vy * tumblescale, // swapped x and y to invert the triangles to face user
                     camFrame.vx * tumblescale,
                     glm::vec3(0, 0, 0),
                     48,
                     color,
                     opacity,
                     code);
  }
  static constexpr uint32_t noCode = -1;
  Gesture::Input::Button& button = gesture.input.mbs[Gesture::Input::kButtonLeft];

  gesture.graphics.addCommand(GL_LINES);

  float axisscale = scale * 0.85f;
  // Draw the x ring in yz plane
  {
    uint32_t code = manipulatorCode(RotateTool::kRotateX, m_codesOffset);
    glm::vec3 color = ManipColors::xAxis;
    if (m_activeCode == RotateTool::kRotateX) {
      color = glm::vec3(1, 1, 0);
    }
    gesture.drawCircle(axis.p, axis.l.vy * axisscale, axis.l.vz * axisscale, 48, color, 1, code);
  }
  // Draw the y ring in xz plane
  {
    uint32_t code = manipulatorCode(RotateTool::kRotateY, m_codesOffset);
    glm::vec3 color = ManipColors::yAxis;
    if (m_activeCode == RotateTool::kRotateY) {
      color = glm::vec3(1, 1, 0);
    }
    gesture.drawCircle(axis.p, axis.l.vz * axisscale, axis.l.vx * axisscale, 48, color, 1, code);
  }
  // Draw the z ring in xy plane
  {
    uint32_t code = manipulatorCode(RotateTool::kRotateZ, m_codesOffset);
    glm::vec3 color = ManipColors::zAxis;
    if (m_activeCode == RotateTool::kRotateZ) {
      color = glm::vec3(1, 1, 0);
      // if we are rotating, draw a tick mark where the rotation started, and where we are now
      // Click in some proportional NDC: x [-1, 1] y [-aspect, aspect]
    }
    gesture.drawCircle(axis.p, axis.l.vx * axisscale, axis.l.vy * axisscale, 48, color, 1, code);
  }

  // Draw the camera-axis rotation manipulator as a circle always facing the view
  {
    uint32_t code = manipulatorCode(RotateTool::kRotateView, m_codesOffset);
    glm::vec3 color = ManipColors::bright;
    if (m_activeCode == RotateTool::kRotateView) {
      color = glm::vec3(1, 1, 0);
    }
    gesture.drawCircle(axis.p, camFrame.vx * scale, camFrame.vy * scale, 48, color, 1, code);
  }

  // if we are rotating, draw a tick mark where the rotation started, and where we are now
  if (button.action == Gesture::Input::Action::kDrag && m_activeCode > -1) {
    // axis.p is the center of a circle in our plane;
    // find the normal of our plane based on active code
    glm::vec3 vN = glm::vec3(0, 0, 0);
    float radius = axisscale;
    if (m_activeCode == RotateTool::kRotateX) {
      vN = axis.l.vx;
    } else if (m_activeCode == RotateTool::kRotateY) {
      vN = axis.l.vy;
    } else if (m_activeCode == RotateTool::kRotateZ) {
      vN = axis.l.vz;
    } else if (m_activeCode == RotateTool::kRotateView) {
      vN = camFrame.vz;
      radius = scale;
    }

    if (vN != glm::vec3(0, 0, 0)) {
      float aperture = scene.camera.getHalfHorizontalAperture();
      glm::vec2 click0 = scene.viewport.toNDC(button.pressedPosition) * aperture;
      glm::vec2 click1 = scene.viewport.toNDC(button.pressedPosition + button.drag) * aperture;
      // l is the camera position, the same "l" in the line parametric eq.
      glm::vec3 l = scene.camera.m_From;
      // l0 is the same "l0" as in the line parametric eq. as the direction of the
      // ray extending into the screen from the position of the initial click.
      // Let's call that "line 0"
      glm::vec3 l0 = normalize(xfmVector(camFrame, glm::vec3(click0.x, click0.y, -1.0)));

      // l1 is same concept as l1 but for the current position of the drag.
      // Let's call that "line 1"
      glm::vec3 l1 = normalize(xfmVector(camFrame, glm::vec3(click1.x, click1.y, -1.0)));

      glm::vec3 color = glm::vec3(1, 1, 0);

      // find nearest point from click position to circle
      glm::vec3 p0 = linePlaneIsect(axis.p, vN, l, l0);
      glm::vec3 p1 = linePlaneIsect(axis.p, vN, l, l1);

      glm::vec3 globalAxis = normalize(axis.p - l);
      bool axisIsOrthogonal = abs(dot(globalAxis, vN)) < s_orthogonalThreshold;
      // need to come up with something different for orthogonal axis
      if (p0 != p1 && !axisIsOrthogonal) {

        // take line back to axis.p
        glm::vec3 v0 = normalize(axis.p - p0);
        glm::vec3 v1 = normalize(axis.p - p1);
        gesture.graphics.addLine(Gesture::Graphics::VertsCode(axis.p - v0 * radius, color, 1, noCode),
                                 Gesture::Graphics::VertsCode(axis.p - v0 * (radius * 1.15f), color, 1, noCode));
        gesture.graphics.addLine(Gesture::Graphics::VertsCode(axis.p - v1 * radius, color, 1, noCode),
                                 Gesture::Graphics::VertsCode(axis.p - v1 * (radius * 1.15f), color, 1, noCode));
        // draw arc showing the rotation
        // start pt, angle, normal
        float a = getSignedAngle(v0, v1, vN);
        gesture.drawArc(axis.p - v0 * (radius * 1.1f),
                        a,
                        axis.p,
                        vN,
                        (int)(96.0 * abs(a) / (glm::two_pi<float>()) + 0.5),
                        color,
                        0.5,
                        noCode);
      }
    }
  }
}
