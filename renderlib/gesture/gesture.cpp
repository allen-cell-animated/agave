#include "gesture.h"

// Update the current action for one of the button of the pointer device
void
Gesture::Input::setButtonEvent(uint32_t mbIndex, Action action, int mods, glm::vec2 position, double time)
{
  if (mbIndex >= kButtonsCount) {
    return;
  }

  Button& button = mbs[mbIndex];
  if (action == Input::kPress) {
    // If the the button is pressed and was previously in a neutral state, it could be a click or a
    // double-click.
    if (button.action == Gesture::Input::kNone) {
      // A double-click event is recorded if the new click follows a previous click of the
      // same button within some time interval.
      button.doubleClick = (time - button.triggerTime) < doubleClickTime;
      button.triggerTime = time;

      // Record position of the pointer during the press event, we are going to use it to
      // determine drag operations
      button.pressedPosition = position;

      // The state of modifiers are recorded when buttons are initially pressed. The state
      // is retained for the duration of the click/drag. We do not update this state until the
      // next button press event.
      button.modifier = mods;
    }

    button.action = Gesture::Input::kPress;
  } else if (action == Input::kRelease) {
    // When button is released, record any drag distance
    if (button.action != Gesture::Input::kNone) {
      glm::vec2 origin = button.pressedPosition;
      button.drag = position - origin;
    }

    button.action = Gesture::Input::kRelease;
  }
}

void
Gesture::Input::setPointerPosition(glm::vec2 position)
{
  cursorPos = position;

  // Update each button action. Each button holding a kPress event becomes a
  // kDrag event.
  for (int mbIndex = 0; mbIndex < Input::kButtonsCount; ++mbIndex) {
    Button& button = mbs[mbIndex];
    if (button.action == Input::kNone || button.action == Input::kRelease) {
      continue;
    }

    glm::vec2 origin = button.pressedPosition;
    glm::vec2 drag = position - origin;
    button.drag = drag;
    bool anyMotion = drag != glm::vec2(0);

    if (button.action == Gesture::Input::kPress && anyMotion) {
      button.action = Gesture::Input::kDrag;

      // If we hold the shift modifier we record if the initial drag is mostly
      // horizontal or vertical. This information may be used by some tools.
      if (button.modifier & Gesture::Input::kShift) {
        button.dragConstraint = (abs(drag.x) > abs(drag.y) ? Gesture::Input::kHorizontal : Gesture::Input::kVertical);
      } else {
        button.dragConstraint = Gesture::Input::kUnconstrained;
      }
    }
  }
}

SceneView::Viewport::Region
SceneView::Viewport::Region::intersect(const SceneView::Viewport::Region& a, const SceneView::Viewport::Region& b)
{
  Region r;
  r.lower.x = std::max(a.lower.x, b.lower.x);
  r.lower.y = std::max(a.lower.y, b.lower.y);
  r.upper.x = std::min(a.upper.x, b.upper.x);
  r.upper.y = std::min(a.upper.y, b.upper.y);
  return r;
}

void
Gesture::drawArc(const glm::vec3& pstart,
                 float angle,
                 const glm::vec3& center,
                 const glm::vec3& normal,
                 uint32_t numSegments,
                 glm::vec3 color,
                 float opacity,
                 uint32_t code)
{
  // draw arc from pstart through angle with center of circle at center
  glm::vec3 xaxis = pstart - center;
  glm::vec3 yaxis = glm::cross(normal, xaxis);
  for (int i = 0; i < numSegments; ++i) {
    float t0 = float(i) / float(numSegments);
    float t1 = float(i + 1) / float(numSegments);

    float theta0 = t0 * angle; // 2.0f * glm::pi<float>();
    float theta1 = t1 * angle; // 2.0f * glm::pi<float>();

    glm::vec3 p0 = center + xaxis * cosf(theta0) + yaxis * sinf(theta0);
    glm::vec3 p1 = center + xaxis * cosf(theta1) + yaxis * sinf(theta1);

    graphics.addLine(Gesture::Graphics::VertsCode(p0, color, opacity, code),
                     Gesture::Graphics::VertsCode(p1, color, opacity, code));
  }
}

void
Gesture::drawArcAsStrip(const glm::vec3& pstart,
                        float angle,
                        const glm::vec3& center,
                        const glm::vec3& normal,
                        uint32_t numSegments,
                        glm::vec3 color,
                        float opacity,
                        uint32_t code)
{
  std::vector<Gesture::Graphics::VertsCode> v;
  // draw arc from pstart through angle with center of circle at center
  glm::vec3 xaxis = pstart - center;
  glm::vec3 yaxis = glm::cross(normal, xaxis);
  for (int i = 0; i < numSegments; ++i) {
    float t0 = float(i) / float(numSegments);
    float t1 = float(i + 1) / float(numSegments);

    float theta0 = t0 * angle; // 2.0f * glm::pi<float>();
    float theta1 = t1 * angle; // 2.0f * glm::pi<float>();

    glm::vec3 p0 = center + xaxis * cosf(theta0) + yaxis * sinf(theta0);
    glm::vec3 p1 = center + xaxis * cosf(theta1) + yaxis * sinf(theta1);

    v.push_back(Gesture::Graphics::VertsCode(p0, color, opacity, code));
    // graphics.addLine(Gesture::Graphics::VertsCode(p0, color, opacity, code),
    //                  Gesture::Graphics::VertsCode(p1, color, opacity, code));
  }
  graphics.addLineStrip(v, 2.0f, false);
}

void
Gesture::drawCircle(glm::vec3 center,
                    glm::vec3 xaxis,
                    glm::vec3 yaxis,
                    uint32_t numSegments,
                    glm::vec3 color,
                    float opacity,
                    uint32_t code,
                    glm::vec4* clipPlane)
{
  for (int i = 0; i < numSegments; ++i) {
    float t0 = float(i) / float(numSegments);
    float t1 = float(i + 1) / float(numSegments);

    float theta0 = t0 * 2.0f * glm::pi<float>();
    float theta1 = t1 * 2.0f * glm::pi<float>();

    glm::vec3 p0 = center + xaxis * cosf(theta0) + yaxis * sinf(theta0);
    glm::vec3 p1 = center + xaxis * cosf(theta1) + yaxis * sinf(theta1);

    if (clipPlane) {
      if (glm::dot(*clipPlane, glm::vec4(p0, 1.0)) > 0 && glm::dot(*clipPlane, glm::vec4(p1, 1.0)) > 0) {
        graphics.addLine(Gesture::Graphics::VertsCode(p0, color, opacity, code),
                         Gesture::Graphics::VertsCode(p1, color, opacity, code));
      }
    } else {
      graphics.addLine(Gesture::Graphics::VertsCode(p0, color, opacity, code),
                       Gesture::Graphics::VertsCode(p1, color, opacity, code));
    }
  }
}

void
Gesture::drawCircleAsStrip(glm::vec3 center,
                           glm::vec3 xaxis,
                           glm::vec3 yaxis,
                           uint32_t numSegments,
                           glm::vec3 color,
                           float opacity,
                           uint32_t code,
                           glm::vec4* clipPlane)
{
  std::vector<Gesture::Graphics::VertsCode> v;
  for (int i = 0; i < numSegments; ++i) {
    float t0 = float(i) / float(numSegments);
    float t1 = float(i + 1) / float(numSegments);

    float theta0 = t0 * 2.0f * glm::pi<float>();
    float theta1 = t1 * 2.0f * glm::pi<float>();

    glm::vec3 p0 = center + xaxis * cosf(theta0) + yaxis * sinf(theta0);
    glm::vec3 p1 = center + xaxis * cosf(theta1) + yaxis * sinf(theta1);

    if (clipPlane) {
      if (glm::dot(*clipPlane, glm::vec4(p0, 1.0)) > 0 && glm::dot(*clipPlane, glm::vec4(p1, 1.0)) > 0) {
        v.push_back(Gesture::Graphics::VertsCode(p0, color, opacity, code));
        // graphics.addLine(Gesture::Graphics::VertsCode(p0, color, opacity, code),
        //                  Gesture::Graphics::VertsCode(p1, color, opacity, code));
      }
    } else {
      v.push_back(Gesture::Graphics::VertsCode(p0, color, opacity, code));
      // graphics.addLine(Gesture::Graphics::VertsCode(p0, color, opacity, code),
      //                  Gesture::Graphics::VertsCode(p1, color, opacity, code));
    }
  }
  graphics.addLineStrip(v, 2.0f, true);
}

// does not draw a flat base
void
Gesture::drawCone(glm::vec3 base,
                  glm::vec3 xaxis,
                  glm::vec3 yaxis,
                  glm::vec3 zaxis,
                  uint32_t numSegments,
                  glm::vec3 color,
                  float opacity,
                  uint32_t code)
{
  for (int i = 0; i < numSegments; ++i) {
    float t0 = float(i) / float(numSegments);
    float t1 = float(i + 1) / float(numSegments);

    float theta0 = t0 * 2.0f * glm::pi<float>();
    float theta1 = t1 * 2.0f * glm::pi<float>();

    glm::vec3 p0 = base + xaxis * cosf(theta0) + yaxis * sinf(theta0);
    glm::vec3 p1 = base + xaxis * cosf(theta1) + yaxis * sinf(theta1);

    graphics.addVert(Gesture::Graphics::VertsCode(base + zaxis, color, opacity, code));
    graphics.addVert(Gesture::Graphics::VertsCode(p1, color, opacity, code));
    graphics.addVert(Gesture::Graphics::VertsCode(p0, color, opacity, code));
  }
}

void
Gesture::drawText(std::string stext, glm::vec3 p, glm::vec2 scale, glm::vec3 color, float opacity, uint32_t code)
{
  float xpos = p.x;
  float ypos = p.y;

  // assume orthographic projection with units = screen pixels, origin at top left
  // also assume we are in a "TRIANGLES" draw command.

  stbtt_aligned_quad q;
  const char* text = stext.c_str();
  while (*text) {
    if (graphics.font.getBakedQuad(*text, &xpos, &ypos, &q)) {
      // apply scaling to q.x0, q.y0, q.x1, q.y1 relative to start position p
      q.x0 = p.x + (q.x0 - p.x) * scale.x;
      q.y0 = p.y + (q.y0 - p.y) * scale.y;
      q.x1 = p.x + (q.x1 - p.x) * scale.x;
      q.y1 = p.y + (q.y1 - p.y) * scale.y;
      // QUAD.
      // 0
      graphics.addVert(
        Gesture::Graphics::VertsCode(glm::vec3(q.x0, q.y0, 0.0), glm::vec2(q.s0, q.t0), color, opacity, code));
      // 2
      graphics.addVert(
        Gesture::Graphics::VertsCode(glm::vec3(q.x1, q.y1, 0.0), glm::vec2(q.s1, q.t1), color, opacity, code));
      // 1
      graphics.addVert(
        Gesture::Graphics::VertsCode(glm::vec3(q.x1, q.y0, 0.0), glm::vec2(q.s1, q.t0), color, opacity, code));

      // 2
      graphics.addVert(
        Gesture::Graphics::VertsCode(glm::vec3(q.x1, q.y1, 0.0), glm::vec2(q.s1, q.t1), color, opacity, code));
      // 0
      graphics.addVert(
        Gesture::Graphics::VertsCode(glm::vec3(q.x0, q.y0, 0.0), glm::vec2(q.s0, q.t0), color, opacity, code));
      // 3
      graphics.addVert(
        Gesture::Graphics::VertsCode(glm::vec3(q.x0, q.y1, 0.0), glm::vec2(q.s0, q.t1), color, opacity, code));
    }
    ++text;
  }
}
