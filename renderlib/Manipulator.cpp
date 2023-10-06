#include "Manipulator.h"

// #include "AppScene.h"

// TODO apply devicePixelRatio to this
float ManipulationTool::s_manipulatorSize = 500.0f;

void
AreaLightTool::action(SceneView& scene, Gesture& gesture)
{
}
void
AreaLightTool::draw(SceneView& scene, Gesture& gesture)
{
  if (!scene.scene) {
    return;
  }
  const Light& l = scene.scene->m_lighting.m_Lights[1];
  glm::vec3 p = l.m_P;
  glm::vec3 t = l.m_Target;
  float scale = l.m_Width * 0.5;
  // compute 4 vertices of square area light pointing at 0, 0, 0
  glm::vec3 v0 = l.m_U * (-scale) + l.m_V * (-scale);
  glm::vec3 v1 = l.m_U * scale + l.m_V * (-scale);
  glm::vec3 v2 = l.m_U * scale + l.m_V * scale;
  glm::vec3 v3 = l.m_U * (-scale) + l.m_V * scale;

  glm::vec3 color = glm::vec3(1, 1, 1);
  float opacity = 1.0f;
  uint32_t code = Gesture::Graphics::SelectionBuffer::k_noSelectionCode;
  gesture.graphics.addCommand(GL_LINES);
  gesture.graphics.addLine(Gesture::Graphics::VertsCode(p + v0, color, opacity, code),
                           Gesture::Graphics::VertsCode(p + v1, color, opacity, code));
  gesture.graphics.extLine(Gesture::Graphics::VertsCode(p + v2, color, opacity, code));
  gesture.graphics.extLine(Gesture::Graphics::VertsCode(p + v3, color, opacity, code));
  gesture.graphics.extLine(Gesture::Graphics::VertsCode(p + v0, color, opacity, code));

  // add translucent quads too?
  opacity = 0.3f;
  gesture.graphics.addCommand(GL_TRIANGLES);
  gesture.graphics.addVert(Gesture::Graphics::VertsCode(p + v0, color, opacity, code));
  gesture.graphics.addVert(Gesture::Graphics::VertsCode(p + v1, color, opacity, code));
  gesture.graphics.addVert(Gesture::Graphics::VertsCode(p + v2, color, opacity, code));
  gesture.graphics.addVert(Gesture::Graphics::VertsCode(p + v0, color, opacity, code));
  gesture.graphics.addVert(Gesture::Graphics::VertsCode(p + v2, color, opacity, code));
  gesture.graphics.addVert(Gesture::Graphics::VertsCode(p + v3, color, opacity, code));

  gesture.graphics.addCommand(GL_LINES);
  gesture.graphics.addLine(Gesture::Graphics::VertsCode(p + v0 * 0.1f, color, opacity, code),
                           Gesture::Graphics::VertsCode(t + v0 * 0.1f, color, opacity, code));
  gesture.graphics.addLine(Gesture::Graphics::VertsCode(p + v1 * 0.1f, color, opacity, code),
                           Gesture::Graphics::VertsCode(t + v1 * 0.1f, color, opacity, code));
  gesture.graphics.addLine(Gesture::Graphics::VertsCode(p + v2 * 0.1f, color, opacity, code),
                           Gesture::Graphics::VertsCode(t + v2 * 0.1f, color, opacity, code));
  gesture.graphics.addLine(Gesture::Graphics::VertsCode(p + v3 * 0.1f, color, opacity, code),
                           Gesture::Graphics::VertsCode(t + v3 * 0.1f, color, opacity, code));

  gesture.graphics.addCommand(GL_TRIANGLES);
  // The cone
  gesture.drawCone(t, l.m_U * scale * 0.2f, l.m_V * scale * 0.2f, l.m_N * (scale * 0.2f), 12, color, opacity, code);

  // The base of the cone (as a flat cone)
  gesture.drawCone(t, l.m_U * scale * 0.2f, l.m_V * scale * 0.2f, glm::vec3(0, 0, 0), 12, color, opacity, code);
}
