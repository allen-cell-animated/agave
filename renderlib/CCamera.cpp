#include "CCamera.h"

#include "gesture/gesture.h"

void
CCamera::SetViewMode(const EViewMode ViewMode)
{
  if (ViewMode == ViewModeUser)
    return;

  glm::vec3 ctr = m_SceneBoundingBox.GetCenter();
  m_Target = ctr;
  m_Up = glm::vec3(0.0f, 1.0f, 0.0f);

  const float size = m_SceneBoundingBox.GetDiagonalLength();
  const float Length = (m_Projection == ORTHOGRAPHIC) ? 2.0f : size * 0.5f / tan(0.5f * m_FovV * DEG_TO_RAD);
  m_OrthoScale = DEF_ORTHO_SCALE;

  // const float Distance = 0.866f;
  // const float Length = Distance * m_SceneBoundingBox.GetMaxLength();

  m_From = m_Target;

  switch (ViewMode) {
    case ViewModeFront:
      m_From.z += Length;
      m_Up = glm::vec3(0.0f, 1.0f, 0.0f);
      break;
    case ViewModeBack:
      m_From.z -= Length;
      m_Up = glm::vec3(0.0f, 1.0f, 0.0f);
      break;
    case ViewModeLeft:
      m_From.x += Length;
      m_Up = glm::vec3(0.0f, 0.0f, 1.0f);
      break;
    case ViewModeRight:
      m_From.x -= Length;
      m_Up = glm::vec3(0.0f, 0.0f, 1.0f);
      break;
    case ViewModeTop:
      m_From.y += Length;
      m_Up = glm::vec3(0.0f, 0.0f, 1.0f);
      break;
    case ViewModeBottom:
      m_From.y -= Length;
      m_Up = glm::vec3(0.0f, 0.0f, 1.0f);
      break;
    case ViewModeIsometricFrontLeftTop:
      m_From = glm::vec3(Length, Length, -Length);
      break;
    case ViewModeIsometricFrontRightTop:
      m_From = m_Target + glm::vec3(-Length, Length, -Length);
      break;
    case ViewModeIsometricFrontLeftBottom:
      m_From = m_Target + glm::vec3(Length, -Length, -Length);
      break;
    case ViewModeIsometricFrontRightBottom:
      m_From = m_Target + glm::vec3(-Length, -Length, -Length);
      break;
    case ViewModeIsometricBackLeftTop:
      m_From = m_Target + glm::vec3(Length, Length, Length);
      break;
    case ViewModeIsometricBackRightTop:
      m_From = m_Target + glm::vec3(-Length, Length, Length);
      break;
    case ViewModeIsometricBackLeftBottom:
      m_From = m_Target + glm::vec3(Length, -Length, Length);
      break;
    case ViewModeIsometricBackRightBottom:
      m_From = m_Target + glm::vec3(-Length, -Length, Length);
      break;
    default:
      break;
  }

  Update();
}

LinearSpace3f
CCamera::getFrame() const
{
  // LOG_DEBUG << "CCamera::getFrame()" << glm::to_string(m_U) << ", " << glm::to_string(m_V) << ", "
  //           << glm::to_string(-m_N) << "\n";

  // see glm::lookat for similar negation of the z vector (opengl convention)
  return LinearSpace3f(m_U, m_V, -m_N);
}

void
CCamera::ComputeFitToBounds(const CBoundingBox& sceneBBox, glm::vec3& newPosition, glm::vec3& newTarget)
{
  // Approximatively compute at what distance the camera need to be to frame
  // the scene content
  glm::vec3 size = sceneBBox.GetExtent();
  glm::vec3 distance = m_From - m_Target;
  float halfWidth = size.length() / 2;
  float halfAperture = getHalfHorizontalAperture();
  const float somePadding = 1.5f;

  distance = glm::normalize(distance) * (somePadding * halfWidth / halfAperture);
  // The new target position is the center of the scene bbox
  newTarget = sceneBBox.GetCenter();
  newPosition = newTarget + distance;
}

glm::vec3
cameraTrack(glm::vec2 drag, CCamera& camera, const glm::vec2 viewportSize)
{
  float width = viewportSize.x;
  glm::vec3 v = camera.m_From - camera.m_Target;
  float distance = length(v);

  // Project the drag movement in pixels to the image plane set at the distance of the
  // camera target.
  float halfHorizontalAperture = camera.getHalfHorizontalAperture();
  float dragScale = distance * halfHorizontalAperture / (width * 0.5f);
  drag *= dragScale;
  glm::vec3 x = glm::normalize(glm::cross(v, camera.m_Up));
  glm::vec3 y = glm::normalize(glm::cross(x, v));
  glm::vec3 track = x * drag.x + y * drag.y;

  return track;
}

bool
cameraManipulationTrack(const glm::vec2 viewportSize,
                        Gesture::Input::Button& button,
                        CCamera& camera,
                        CameraModifier& cameraMod)
{
  bool cameraEdit = true;
  glm::vec2 drag = button.drag;
  if (button.dragConstraint == Gesture::Input::kHorizontal) {
    drag.y = 0;
  }
  if (button.dragConstraint == Gesture::Input::kVertical) {
    drag.x = 0;
  }

  if (drag.x == 0 && drag.y == 0) {
    cameraEdit = false;
  }

  glm::vec3 track = cameraTrack(drag, camera, viewportSize);

  if (button.action == Gesture::Input::kPress || button.action == Gesture::Input::kDrag) {
    cameraMod.position = track;
    cameraMod.target = track;
  } else if (button.action == Gesture::Input::kRelease) {
    camera.m_From += track;
    camera.m_Target += track;

    // Consume pointer button release event
    Gesture::Input::reset(button);
  }
  return cameraEdit;
}
bool
cameraManipulationDolly(const glm::vec2 viewportSize,
                        Gesture::Input::Button& button,
                        CCamera& camera,
                        CameraModifier& cameraMod)
{

  bool cameraEdit = true;
  static const int DOLLY_PIXELS_PER_UNIT = 700;
  const float dragScale = 1.0f / DOLLY_PIXELS_PER_UNIT;

  glm::vec2 drag = button.drag;
  float dragDist = drag.x + drag.y; // reduce_add(drag);
  glm::vec3 v = camera.m_From - camera.m_Target;
  glm::vec3 motion, targetMotion;

  if (drag.x == 0 && drag.y == 0) {
    cameraEdit = false;
  }

  float factor = 1.0f;
  if ((button.modifier & Gesture::Input::kShift) == 0) {
    // Exponential motion, the closer to the target, the slower the motion,
    // the further away the faster.
    factor = expf(-dragDist * dragScale);
    glm::vec3 v_scaled = v * factor;
    motion = v_scaled - v;
    targetMotion = glm::vec3(0);
  } else {
    // Linear motion. We move position and target at once. This mode allows the user not
    // to get stuck with a camera that doesn't move because the position got too close to
    // the target.
    factor = (-dragDist * dragScale);
    glm::vec3 v_scaled = v * factor;
    motion = v_scaled;
    targetMotion = motion;
  }

  if (button.action == Gesture::Input::kPress || button.action == Gesture::Input::kDrag) {
    cameraMod.position = motion;
    cameraMod.target = targetMotion;
  } else if (button.action == Gesture::Input::kRelease) {
    camera.m_From += motion;
    camera.m_Target += targetMotion;
    camera.m_OrthoScale *= factor;
    camera.Update();

    // Consume gesture on button release
    Gesture::Input::reset(button);
  }
  return cameraEdit;
}

bool
cameraManipulation(const glm::vec2 viewportSize, Gesture& gesture, CCamera& camera, CameraModifier& cameraMod)
{
  float width = viewportSize.x;
  bool cameraEdit = false;

  // Camera Track
  // middle-drag or alt-left-drag (option-left-drag on Mac)
  if (gesture.input.hasButtonAction(Gesture::Input::kButtonMiddle, 0)) {
    cameraEdit =
      cameraManipulationTrack(viewportSize, gesture.input.mbs[Gesture::Input::kButtonMiddle], camera, cameraMod);
  } else if (gesture.input.hasButtonAction(Gesture::Input::kButtonLeft, Gesture::Input::kAlt)) {
    cameraEdit =
      cameraManipulationTrack(viewportSize, gesture.input.mbs[Gesture::Input::kButtonLeft], camera, cameraMod);
  }

  // Camera Roll
  else if (gesture.input.hasButtonAction(Gesture::Input::kButtonMiddle, Gesture::Input::kCtrl)) {
    cameraEdit = true;
    Gesture::Input::Button& button = gesture.input.mbs[Gesture::Input::kButtonMiddle];

    static const int ROLL_PIXELS_PER_RADIAN = 400;
    const float dragScale = -1.0f / ROLL_PIXELS_PER_RADIAN;

    glm::vec2 drag = button.drag * dragScale;
    glm::vec3 up = glm::normalize(camera.m_Up);
    glm::vec3 v = glm::normalize(camera.m_From - camera.m_Target);

    if (drag.x == 0 && drag.y == 0)
      cameraEdit = false;

    // Rotate the up vector around the camera to target direction
    glm::vec3 rotated_up =
      glm::rotate(camera.m_Up, drag.x, v); // xfmVector(Quaternion3f::rotate(v, drag.x), camera.up);

    if (button.action == Gesture::Input::kPress || button.action == Gesture::Input::kDrag) {
      cameraMod.up = rotated_up - camera.m_Up;
    } else if (button.action == Gesture::Input::kRelease) {
      camera.m_Up = rotated_up;
      cameraMod.position = glm::vec3(0);

      // Consume pointer button release event
      Gesture::Input::reset(button);
    }
  }

  // Camera Dolly
  // right-drag or ctrl-left-drag (command-left-drag on Mac)
  else if (gesture.input.hasButtonAction(Gesture::Input::kButtonRight, 0)) {
    cameraEdit =
      cameraManipulationDolly(viewportSize, gesture.input.mbs[Gesture::Input::kButtonRight], camera, cameraMod);
  } else if (gesture.input.hasButtonAction(Gesture::Input::kButtonLeft, Gesture::Input::kCtrl)) {
    cameraEdit =
      cameraManipulationDolly(viewportSize, gesture.input.mbs[Gesture::Input::kButtonLeft], camera, cameraMod);
  }
  // Camera Tumble
  // hasButtonAction with zero actions must be last to check for, since it will match anything
  else if (gesture.input.hasButtonAction(Gesture::Input::kButtonLeft, 0 /*Gesture::Input::kAlt*/)) {
    cameraEdit = true;
    Gesture::Input::Button& button = gesture.input.mbs[Gesture::Input::kButtonLeft];

    static const int TUMBLE_PIXELS_PER_RADIAN = 800;
    const float dragScale = 1.0f / TUMBLE_PIXELS_PER_RADIAN;

    glm::vec2 drag = button.drag * dragScale;
    if (button.dragConstraint == Gesture::Input::kHorizontal) {
      drag.y = 0;
    }
    if (button.dragConstraint == Gesture::Input::kVertical) {
      drag.x = 0;
    }

    if (drag.x == 0 && drag.y == 0) {
      cameraEdit = false;
    }

    glm::vec3 v = camera.m_From - camera.m_Target;
    glm::quat q = trackball(drag.x, -drag.y, v, camera.m_Up, camera.m_U);
    glm::vec3 rotated_up = q * camera.m_Up;
    glm::vec3 rotated_v = q * v;

    if (button.action == Gesture::Input::kPress || button.action == Gesture::Input::kDrag) {
      cameraMod.position = rotated_v - v;
      cameraMod.up = rotated_up - camera.m_Up;
    } else if (button.action == Gesture::Input::kRelease) {
      camera.m_From += rotated_v - v;
      camera.m_Up = rotated_up;
      camera.Update();

      // Consume gesture on button release
      Gesture::Input::reset(button);
    }
  }

  return cameraEdit;
}
