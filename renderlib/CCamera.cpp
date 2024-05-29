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
#if 0
// Returns whether or not the given point is the outermost point in the given direction among all points of the bounds
private
static bool
IsOutermostPointInDirection(int pointIndex, Vector3 direction)
{
  Vector3 point = boundingBoxPoints[pointIndex];
  for (int i = 0; i < boundingBoxPoints.Length; i++) {
    if (i != pointIndex && Vector3.Dot(direction, boundingBoxPoints[i] - point) > 0)
      return false;
  }

  return true;
}

// Credit: http://wiki.unity3d.com/index.php/3d_Math_functions
// Returns the edge points of the closest line segment between 2 lines
private
static void
FindClosestPointsOnTwoLines(Ray line1, Ray line2, out Vector3 closestPointLine1, out Vector3 closestPointLine2)
{
  Vector3 line1Direction = line1.direction;
  Vector3 line2Direction = line2.direction;

  float a = Vector3.Dot(line1Direction, line1Direction);
  float b = Vector3.Dot(line1Direction, line2Direction);
  float e = Vector3.Dot(line2Direction, line2Direction);

  float d = a * e - b * b;

  Vector3 r = line1.origin - line2.origin;
  float c = Vector3.Dot(line1Direction, r);
  float f = Vector3.Dot(line2Direction, r);

  float s = (b * f - c * e) / d;
  float t = (a * f - c * b) / d;

  closestPointLine1 = line1.origin + line1Direction * s;
  closestPointLine2 = line2.origin + line2Direction * t;
}
// Credit: https://stackoverflow.com/a/32410473/2373034
// Returns the intersection line of the 2 planes
private
static Ray
GetPlanesIntersection(Plane p1, Plane p2)
{
  Vector3 p3Normal = Vector3.Cross(p1.normal, p2.normal);
  float det = p3Normal.sqrMagnitude;

  return new Ray(
    ((Vector3.Cross(p3Normal, p2.normal) * p1.distance) + (Vector3.Cross(p1.normal, p3Normal) * p2.distance)) / det,
    p3Normal);
}
#endif
void
CCamera::ComputeFitToBounds(const CBoundingBox& sceneBBox, glm::vec3& newPosition, glm::vec3& newTarget)
{
#if 0
  if (m_Projection == ORTHOGRAPHIC) {
    newPosition = sceneBBox.GetCenter();
    cameraTR.position = boundsCenter;

    float minX = float.PositiveInfinity, minY = float.PositiveInfinity;
    float maxX = float.NegativeInfinity, maxY = float.NegativeInfinity;

    for (int i = 0; i < boundingBoxPoints.Length; i++) {
      Vector3 localPoint = cameraTR.InverseTransformPoint(boundingBoxPoints[i]);
      if (localPoint.x < minX)
        minX = localPoint.x;
      if (localPoint.x > maxX)
        maxX = localPoint.x;
      if (localPoint.y < minY)
        minY = localPoint.y;
      if (localPoint.y > maxY)
        maxY = localPoint.y;
    }

    float distance = boundsExtents.magnitude + 1f;
    camera.orthographicSize = Mathf.Max(maxY - minY, (maxX - minX) / aspect) * 0.5f;
    cameraTR.position = boundsCenter - cameraDirection * distance;
  } else {
    Vector3 cameraUp = cameraTR.up, cameraRight = cameraTR.right;

    float verticalFOV = camera.fieldOfView * 0.5f;
    float horizontalFOV = Mathf.Atan(Mathf.Tan(verticalFOV * Mathf.Deg2Rad) * aspect) * Mathf.Rad2Deg;

    // Normals of the camera's frustum planes
    Vector3 topFrustumPlaneNormal = Quaternion.AngleAxis(90f + verticalFOV, -cameraRight) * cameraDirection;
    Vector3 bottomFrustumPlaneNormal = Quaternion.AngleAxis(90f + verticalFOV, cameraRight) * cameraDirection;
    Vector3 rightFrustumPlaneNormal = Quaternion.AngleAxis(90f + horizontalFOV, cameraUp) * cameraDirection;
    Vector3 leftFrustumPlaneNormal = Quaternion.AngleAxis(90f + horizontalFOV, -cameraUp) * cameraDirection;

    // Credit for algorithm: https://stackoverflow.com/a/66113254/2373034
    // 1. Find edge points of the bounds using the camera's frustum planes
    // 2. Create a plane for each edge point that goes through the point and has the corresponding frustum plane's
    // normal
    // 3. Find the intersection line of horizontal edge points' planes (horizontalIntersection) and vertical edge
    // points' planes (verticalIntersection)
    //    If we move the camera along horizontalIntersection, the bounds will always with the camera's width perfectly
    //    (similar effect goes for verticalIntersection)
    // 4. Find the closest line segment between these two lines (horizontalIntersection and verticalIntersection) and
    // place the camera at the farthest point on that line
    int leftmostPoint = -1, rightmostPoint = -1, topmostPoint = -1, bottommostPoint = -1;
    for (int i = 0; i < boundingBoxPoints.Length; i++) {
      if (leftmostPoint < 0 && IsOutermostPointInDirection(i, leftFrustumPlaneNormal))
        leftmostPoint = i;
      if (rightmostPoint < 0 && IsOutermostPointInDirection(i, rightFrustumPlaneNormal))
        rightmostPoint = i;
      if (topmostPoint < 0 && IsOutermostPointInDirection(i, topFrustumPlaneNormal))
        topmostPoint = i;
      if (bottommostPoint < 0 && IsOutermostPointInDirection(i, bottomFrustumPlaneNormal))
        bottommostPoint = i;
    }

    Ray horizontalIntersection =
      GetPlanesIntersection(new Plane(leftFrustumPlaneNormal, boundingBoxPoints[leftmostPoint]),
                            new Plane(rightFrustumPlaneNormal, boundingBoxPoints[rightmostPoint]));
    Ray verticalIntersection =
      GetPlanesIntersection(new Plane(topFrustumPlaneNormal, boundingBoxPoints[topmostPoint]),
                            new Plane(bottomFrustumPlaneNormal, boundingBoxPoints[bottommostPoint]));

    Vector3 closestPoint1, closestPoint2;
    FindClosestPointsOnTwoLines(horizontalIntersection, verticalIntersection, out closestPoint1, out closestPoint2);

    cameraTR.position = Vector3.Dot(closestPoint1 - closestPoint2, cameraDirection) < 0 ? closestPoint1 : closestPoint2;
  }
}
#endif
  glm::mat4 mv, mp;
  getViewMatrix(mv);
  getProjMatrix(mp);
  glm::mat4 mvp = mp * mv;
  CRegion region = sceneBBox.projectToXY(mvp);
  glm::vec2 size = region.size();
  float maxdim = std::max(size.x, size.y);
  float ratio = maxdim / 2.0f;

  glm::vec3 direction = m_From - m_Target;
  direction = direction * ratio;
  // The new target position is the center of the scene bbox
  newTarget = sceneBBox.GetCenter();
  newPosition = newTarget + direction;

  // the center is a vector in clip space that we need to transform back to world space.
  // glm::vec4 centernear = glm::vec4(region.center(), 0.0f, 1.0f);
  // glm::vec4 centerfar = glm::vec4(region.center(), 1.0f, 1.0f);
  // glm::vec4 centerWorldNear = glm::inverse(mvp) * centernear;
  // glm::vec4 centerWorldFar = glm::inverse(mvp) * centerfar;
#if 0

  // now we have a region in clip space.
  // the ratio of this region is proportional to the ratio of the camera distances?

  // Approximatively compute at what distance the camera need to be to frame
  // the scene content
  float boundingSphereRadius = sceneBBox.GetEquivalentRadius();

  glm::vec3 direction = m_From - m_Target;
  float halfAperture = getHalfHorizontalAperture();
  const float somePadding = 1.1f;
  // the distance at which the bounding sphere fits in the FOV
  float distanceFit = boundingSphereRadius / halfAperture;

  direction = glm::normalize(direction) * (somePadding * distanceFit);
  // The new target position is the center of the scene bbox
  newTarget = sceneBBox.GetCenter();
  newPosition = newTarget + direction;
#endif
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
