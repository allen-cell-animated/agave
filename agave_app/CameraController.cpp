#include "CameraController.h"

#include "RenderSettings.h"

// renderlib
#include "renderlib/CCamera.h"
#include "renderlib/Logging.h"
#include "renderlib/gesture/gesture.h"

#include <QApplication>
#include <QGuiApplication>
#include <QMouseEvent>
#include <QWindow>

float CameraController::m_OrbitSpeed = 1.0f;
float CameraController::m_PanSpeed = 1.0f;
float CameraController::m_ZoomSpeed = 1000.0f;
float CameraController::m_ContinuousZoomSpeed = 0.0000001f;
float CameraController::m_ApertureSpeed = 0.001f;
float CameraController::m_FovSpeed = 0.5f;

CameraController::CameraController(QCamera* cam, CCamera* theCamera)
  : m_renderSettings(nullptr)
  , m_qcamera(cam)
  , m_CCamera(theCamera)
{
}

void
CameraController::OnMouseWheelForward(void)
{
  m_CCamera->Zoom(-m_ZoomSpeed);

  // Flag the camera as dirty, this will restart the rendering
  // m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
};

void
CameraController::OnMouseWheelBackward(void)
{
  m_CCamera->Zoom(m_ZoomSpeed);

  // Flag the camera as dirty, this will restart the rendering
  // m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
};

bool
GetShiftKey()
{
  return QGuiApplication::keyboardModifiers() & Qt::ShiftModifier;
}
bool
GetCtrlKey()
{
  return QGuiApplication::keyboardModifiers() & Qt::ControlModifier;
}
bool
GetAltKey()
{
  return QGuiApplication::keyboardModifiers() & Qt::AltModifier;
}

void
CameraController::OnMouseMove(QMouseEvent* event)
{

  float devicePixelRatio = QGuiApplication::focusWindow()->devicePixelRatio();
  if (event->buttons() & Qt::LeftButton) {
    if (GetCtrlKey()) {
      // Zooming (Dolly): ctrl + left drag
      m_NewPos[0] = event->x();
      m_NewPos[1] = event->y();

      m_CCamera->Zoom(-(float)(m_NewPos[1] - m_OldPos[1]) / devicePixelRatio);

      m_OldPos[0] = event->x();
      m_OldPos[1] = event->y();

      // Flag the camera as dirty, this will restart the rendering
      // m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
    } else if (GetAltKey()) {
      // Panning (tracking): alt + left drag
      m_NewPos[0] = event->x();
      m_NewPos[1] = event->y();

      m_CCamera->Pan(m_PanSpeed * (float)(m_NewPos[1] - m_OldPos[1]) / devicePixelRatio,
                     -m_PanSpeed * ((float)(m_NewPos[0] - m_OldPos[0]) / devicePixelRatio));

      m_OldPos[0] = event->x();
      m_OldPos[1] = event->y();

      // Flag the camera as dirty, this will restart the rendering
      // m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
    } else {
      // Orbiting: unmodified left button

      m_NewPos[0] = event->x();
      m_NewPos[1] = event->y();

      if (m_NewPos[1] != m_OldPos[1] || m_NewPos[0] != m_OldPos[0]) {
        m_CCamera->Trackball(-m_OrbitSpeed * (float)(m_NewPos[1] - m_OldPos[1]) / devicePixelRatio,
                             -m_OrbitSpeed * (float)(m_NewPos[0] - m_OldPos[0]) / devicePixelRatio);

        m_OldPos[0] = event->x();
        m_OldPos[1] = event->y();

        // Flag the camera as dirty, this will restart the rendering
        // m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
      }
    }
  }

  // Panning (tracking): middle mouse drag
  if (event->buttons() & Qt::MiddleButton) {
    m_NewPos[0] = event->x();
    m_NewPos[1] = event->y();

    m_CCamera->Pan(m_PanSpeed * (float)(m_NewPos[1] - m_OldPos[1]) / devicePixelRatio,
                   -m_PanSpeed * ((float)(m_NewPos[0] - m_OldPos[0]) / devicePixelRatio));

    m_OldPos[0] = event->x();
    m_OldPos[1] = event->y();

    // Flag the camera as dirty, this will restart the rendering
    // m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
  }

  // Zooming (Dolly): right mouse drag
  if (event->buttons() & Qt::RightButton) {
    m_NewPos[0] = event->x();
    m_NewPos[1] = event->y();

    m_CCamera->Zoom(-(float)(m_NewPos[1] - m_OldPos[1]) / devicePixelRatio);

    m_OldPos[0] = event->x();
    m_OldPos[1] = event->y();

    // Flag the camera as dirty, this will restart the rendering
    // m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
  }
}

// keep target constant and compute new eye and up vectors
void
trackball(float xRadians, float yRadians, const CCamera& camera, glm::vec3& eye, glm::vec3& up)
{
  float angle = sqrtf(yRadians * yRadians + xRadians * xRadians);
  if (angle == 0.0f) {
    // skip some extra math
    eye = camera.m_From;
    up = camera.m_Up;
    return;
  }
  glm::vec3 _eye = camera.m_From - camera.m_Target;

  glm::vec3 objectUpDirection = camera.m_Up; // or m_V; ???
  glm::vec3 objectSidewaysDirection = camera.m_U;

  objectUpDirection *= yRadians;
  objectSidewaysDirection *= -xRadians;

  glm::vec3 moveDirection = objectUpDirection + objectSidewaysDirection;

  glm::vec3 axis = glm::normalize(glm::cross(moveDirection, _eye));

  _eye = glm::rotate(_eye, angle, axis);

  up = glm::rotate(camera.m_Up, angle, axis);
  eye = _eye + camera.m_Target;
}

bool
cameraManipulation(const glm::vec2 viewportSize,
                   // const TimeSample& clock,
                   Gesture& gesture,
                   CCamera& camera,
                   CameraModifier& cameraMod)
{
  float width = viewportSize.x;
  bool cameraEdit = false;

  // Camera Track
  if (gesture.input.hasButtonAction(Gesture::Input::kButtonLeft, Gesture::Input::kAlt)) {
    cameraEdit = true;
    Gesture::Input::Button& button = gesture.input.mbs[Gesture::Input::kButtonLeft];

    glm::vec2 drag = button.drag;
    if (button.dragConstraint == Gesture::Input::kHorizontal) {
      drag.y = 0;
    }
    if (button.dragConstraint == Gesture::Input::kVertical) {
      drag.x = 0;
    }

    glm::vec3 v = camera.m_From - camera.m_Target;
    float distance = length(v);

    if (drag.x == 0 && drag.y == 0) {
      cameraEdit = false;
    }

// Project the drag movement in pixels to the image plane set at the distance of the
// camera target.
#define deg2rad(x) ((x)*0.01745329251f)
    float halfHorizontalAperture = tanf(deg2rad(camera.GetHorizontalFOV()) * 0.5f);
    float dragScale = distance * halfHorizontalAperture / (width * 0.5f);
    drag *= dragScale;
    glm::vec3 x = glm::normalize(glm::cross(v, camera.m_Up));
    glm::vec3 y = glm::normalize(glm::cross(x, v));
    glm::vec3 track = x * drag.x + y * drag.y;

    if (button.action == Gesture::Input::kPress || button.action == Gesture::Input::kDrag) {
      cameraMod.position = track;
      cameraMod.target = track;
    } else if (button.action == Gesture::Input::kRelease) {
      camera.m_From += track;
      camera.m_Target += track;

      // Consume pointer button release event
      Gesture::Input::reset(button);
    }
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
  else if (gesture.input.hasButtonAction(Gesture::Input::kButtonLeft, Gesture::Input::kCtrl)) {
    cameraEdit = true;
    Gesture::Input::Button& button = gesture.input.mbs[Gesture::Input::kButtonLeft];

    static const int DOLLY_PIXELS_PER_UNIT = 700;
    const float dragScale = 1.0f / DOLLY_PIXELS_PER_UNIT;

    glm::vec2 drag = button.drag;
    float dragDist = drag.x + drag.y; // reduce_add(drag);
    glm::vec3 v = camera.m_From - camera.m_Target;
    glm::vec3 motion, targetMotion;

    if (drag.x == 0 && drag.y == 0)
      cameraEdit = false;

    if ((button.modifier & Gesture::Input::kShift) == 0) {
      // Exponential motion, the closer to the target, the slower the motion,
      // the further away the faster.
      glm::vec3 v_scaled = v * expf(-dragDist * dragScale);
      motion = v_scaled - v;
      targetMotion = glm::vec3(0);
    } else {
      // Linear motion. We move position and target at once. This mode allows the user not
      // to get stuck with a camera that doesn't move because the postion got to close to
      // the target.
      glm::vec3 v_scaled = v * (-dragDist * dragScale);
      motion = v_scaled;
      targetMotion = motion;
    }

    if (button.action == Gesture::Input::kPress || button.action == Gesture::Input::kDrag) {
      cameraMod.position = motion;
      cameraMod.target = targetMotion;
    } else if (button.action == Gesture::Input::kRelease) {
      camera.m_From += motion;
      camera.m_Target += targetMotion;

      // Consume gesture on button release
      Gesture::Input::reset(button);
    }
  }
  // Camera Tumble
  // hasButtonAction with zero actions must be last to check for, since it will match anything
  else if (gesture.input.hasButtonAction(Gesture::Input::kButtonLeft, 0 /*Gesture::Input::kAlt*/)) {
    cameraEdit = true;
    Gesture::Input::Button& button = gesture.input.mbs[Gesture::Input::kButtonLeft];

    static const int TUMBLE_PIXELS_PER_RADIAN = 400;
    const float dragScale = -1.0f / TUMBLE_PIXELS_PER_RADIAN;

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

    glm::vec3 rotated_up = camera.m_Up;
    glm::vec3 rotated_v = camera.m_From - camera.m_Target;
    glm::vec3 v = camera.m_From - camera.m_Target;
    glm::vec3 newEye, newUp;
    trackball(drag.x, drag.y, camera, newEye, newUp);
    rotated_v = newEye - camera.m_Target;
    rotated_up = newUp;

#if 0
    // First rotation is horizontal, around the up vector.
    glm::vec3 v = camera.m_From - camera.m_Target;
    glm::vec3 rotated_v =
      glm::rotate(v, drag.x, camera.m_Up); // xfmVector(Quaternion3f::rotate(camera.m_Up, drag.x), v);

    // Second rotation is vertical, around the camera x-axis.
    glm::vec3 x_axis = glm::normalize(glm::cross(camera.m_Up, rotated_v));
    rotated_v = glm::rotate(rotated_v, drag.y, x_axis); // xfmVector(Quaternion3f::rotate(x_axis, drag.y), rotated_v);

    // Check if after vertical rotation the camera swang to the other side of the up vector.
    // When this happens we must flip the up vector to give the user a continuous rotation.
    bool flipped = glm::dot(x_axis, glm::cross(camera.m_Up, rotated_v)) < 0;
    glm::vec3 rotated_up = flipped ? -camera.m_Up : camera.m_Up;
#endif

    if (button.action == Gesture::Input::kPress || button.action == Gesture::Input::kDrag) {
      cameraMod.position = rotated_v - v;
      cameraMod.up = rotated_up - camera.m_Up;
    } else if (button.action == Gesture::Input::kRelease) {
      camera.m_From += rotated_v - v;
      camera.m_Up = rotated_up;

      // Consume gesture on button release
      Gesture::Input::reset(button);
    }
  }

  return cameraEdit;
}
