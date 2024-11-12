#include "ClipPlaneTool.h"

#include "Logging.h"

void
ClipPlaneTool::action(SceneView& scene, Gesture& gesture)
{
}
void
ClipPlaneTool::draw(SceneView& scene, Gesture& gesture)
{
  if (!scene.scene) {
    return;
  }
  // TODO draw this as an oriented grid centered in the view (or on the volume?)
  glm::vec3 n = m_plane.normal;
  glm::vec3 p = m_plane.normal * m_plane.d;
  glm::vec3 u = glm::cross(n, glm::vec3(0, 1, 0));
  if (glm::length(u) < 0.001f) {
    u = glm::normalize(glm::cross(n, glm::vec3(1, 0, 0)));
  } else {
    u = glm::normalize(u);
  }
  glm::vec3 v = glm::normalize(glm::cross(n, u));
  u = glm::normalize(glm::cross(v, n));

  glm::vec3 color = glm::vec3(1, 1, 1);
  float opacity = 1.0f;
  uint32_t code = Gesture::Graphics::k_noSelectionCode;
  gesture.graphics.addCommand(Gesture::Graphics::PrimitiveType::kLines);

  constexpr int numLines = 4;
  float scale = 0.5f;
  // draw a grid across the u and v axes centered at p
  for (int i = 0; i < numLines + 1; ++i) {
    float fraction = ((float)i - numLines / 2.0f) / (float)numLines;
    glm::vec3 p0 = p + u * (fraction * scale) + v * (0.5f * scale);
    glm::vec3 p1 = p + u * (fraction * scale) + v * (-0.5f * scale);
    gesture.graphics.addLine(Gesture::Graphics::VertsCode(p0, color, opacity, code),
                             Gesture::Graphics::VertsCode(p1, color, opacity, code));
    p0 = p + u * (0.5f * scale) + v * (fraction * scale);
    p1 = p + u * (-0.5f * scale) + v * (fraction * scale);
    gesture.graphics.addLine(Gesture::Graphics::VertsCode(p0, color, opacity, code),
                             Gesture::Graphics::VertsCode(p1, color, opacity, code));
  }

  gesture.graphics.addCommand(Gesture::Graphics::PrimitiveType::kTriangles);
  // fill in the rectangle of the plane spanning the u and v axes
  glm::vec3 p0 = p - u * scale - v * scale;
  glm::vec3 p1 = p + u * scale - v * scale;
  glm::vec3 p2 = p + u * scale + v * scale;
  glm::vec3 p3 = p - u * scale + v * scale;
  gesture.graphics.addVert(Gesture::Graphics::VertsCode(p0, color, opacity, code));
  gesture.graphics.addVert(Gesture::Graphics::VertsCode(p1, color, opacity, code));
  gesture.graphics.addVert(Gesture::Graphics::VertsCode(p2, color, opacity, code));
  gesture.graphics.addVert(Gesture::Graphics::VertsCode(p0, color, opacity, code));
  gesture.graphics.addVert(Gesture::Graphics::VertsCode(p2, color, opacity, code));
  gesture.graphics.addVert(Gesture::Graphics::VertsCode(p3, color, opacity, code));
}