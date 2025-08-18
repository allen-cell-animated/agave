#include "BoundingBoxTool.h"

#include "AppScene.h"
#include "BoundingBox.h"
#include "MathUtil.h"

static const float s_lineThickness = 4.0f;

void
BoundingBoxTool::action(SceneView& scene, Gesture& gesture)
{
  // BoundingBox tool is primarily for visualization, no interactive actions needed
}

void
BoundingBoxTool::draw(SceneView& scene, Gesture& gesture)
{
  const Scene* theScene = scene.scene;
  if (!theScene) {
    return;
  }

  if (!theScene->m_material.m_showBoundingBox) {
    return;
  }

  const CBoundingBox& bbox = theScene->m_boundingBox;

  // Don't draw if bounding box is invalid
  if (bbox.m_MinP.x >= bbox.m_MaxP.x || bbox.m_MinP.y >= bbox.m_MaxP.y || bbox.m_MinP.z >= bbox.m_MaxP.z) {
    return;
  }

  glm::vec3 center = bbox.GetCenter();
  glm::vec3 extent = bbox.GetExtent();
  glm::vec3 halfExtent = extent * 0.5f;

  // Calculate the 8 corners of the bounding box
  std::array<glm::vec3, 8> corners = {
    center + glm::vec3(-halfExtent.x, -halfExtent.y, -halfExtent.z), // 0: min corner
    center + glm::vec3(halfExtent.x, -halfExtent.y, -halfExtent.z),  // 1
    center + glm::vec3(halfExtent.x, halfExtent.y, -halfExtent.z),   // 2
    center + glm::vec3(-halfExtent.x, halfExtent.y, -halfExtent.z),  // 3
    center + glm::vec3(-halfExtent.x, -halfExtent.y, halfExtent.z),  // 4
    center + glm::vec3(halfExtent.x, -halfExtent.y, halfExtent.z),   // 5
    center + glm::vec3(halfExtent.x, halfExtent.y, halfExtent.z),    // 6: max corner
    center + glm::vec3(-halfExtent.x, halfExtent.y, halfExtent.z)    // 7
  };

  // Get bounding box color from scene material
  glm::vec3 color = glm::vec3(theScene->m_material.m_boundingBoxColor[0],
                              theScene->m_material.m_boundingBoxColor[1],
                              theScene->m_material.m_boundingBoxColor[2]);
  color = glm::vec3(1, 0, 0);

  float opacity = 1.0f;
  uint32_t code = Gesture::Graphics::k_noSelectionCode;

  // Draw the 12 edges of the bounding box using addLineStrip for thick lines
  // Bottom face edges (closed loop)
  gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(corners[0], color, opacity, code),
                                  Gesture::Graphics::VertsCode(corners[1], color, opacity, code),
                                  Gesture::Graphics::VertsCode(corners[2], color, opacity, code),
                                  Gesture::Graphics::VertsCode(corners[3], color, opacity, code),
                                  Gesture::Graphics::VertsCode(corners[0], color, opacity, code) },
                                s_lineThickness,
                                true);

  // Top face edges (closed loop)
  gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(corners[4], color, opacity, code),
                                  Gesture::Graphics::VertsCode(corners[5], color, opacity, code),
                                  Gesture::Graphics::VertsCode(corners[6], color, opacity, code),
                                  Gesture::Graphics::VertsCode(corners[7], color, opacity, code),
                                  Gesture::Graphics::VertsCode(corners[4], color, opacity, code) },
                                s_lineThickness,
                                true);

  // Vertical edges connecting bottom and top
  gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(corners[0], color, opacity, code),
                                  Gesture::Graphics::VertsCode(corners[4], color, opacity, code) },
                                s_lineThickness);
  gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(corners[1], color, opacity, code),
                                  Gesture::Graphics::VertsCode(corners[5], color, opacity, code) },
                                s_lineThickness);
  gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(corners[2], color, opacity, code),
                                  Gesture::Graphics::VertsCode(corners[6], color, opacity, code) },
                                s_lineThickness);
  gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(corners[3], color, opacity, code),
                                  Gesture::Graphics::VertsCode(corners[7], color, opacity, code) },
                                s_lineThickness);

  if (theScene->m_showScaleBar && scene.camera.m_Projection != ProjectionMode::ORTHOGRAPHIC) {
    // Draw tick marks similar to BoundingBoxDrawable
    drawTickMarks(bbox, gesture, color, opacity, code);
  }
}

void
BoundingBoxTool::drawTickMarks(const CBoundingBox& bbox,
                               Gesture& gesture,
                               const glm::vec3& color,
                               float opacity,
                               uint32_t code)
{
  glm::vec3 center = bbox.GetCenter();
  glm::vec3 extent = bbox.GetExtent();
  glm::vec3 halfExtent = extent * 0.5f;

  // Add thick line command for tick marks
  // Note: addLineStrip is self-contained and doesn't require addCommand

  // Length of tick mark lines as a fraction of the smallest dimension
  float minDim = glm::min(glm::min(extent.x, extent.y), extent.z);
  float tickLength = minDim * 0.025f; // 2.5% of smallest dimension

  // Calculate physical scale based on extent - use largest dimension
  float maxDim = glm::max(glm::max(extent.x, extent.y), extent.z);
  float tickSpacing = computePhysicalScaleBarSize(maxDim) / maxDim; // Normalized tick spacing

  // Ensure we don't create too many or too few tick marks
  tickSpacing = glm::max(tickSpacing, 0.1f); // At least 10 ticks max
  tickSpacing = glm::min(tickSpacing, 0.5f); // At most 2 ticks per dimension

  // Draw tick marks along X axis on the bottom edges
  for (float t = 0.0f; t <= 1.0f; t += tickSpacing) {
    if (t > 1.0f)
      t = 1.0f;

    float x = center.x + (t - 0.5f) * extent.x;

    // Bottom front edge
    glm::vec3 p1(x, bbox.m_MinP.y, bbox.m_MaxP.z);
    glm::vec3 p2(x, bbox.m_MinP.y - tickLength, bbox.m_MaxP.z);
    gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(p1, color, opacity, code),
                                    Gesture::Graphics::VertsCode(p2, color, opacity, code) },
                                  s_lineThickness);

    // Bottom back edge
    p1 = glm::vec3(x, bbox.m_MinP.y, bbox.m_MinP.z);
    p2 = glm::vec3(x, bbox.m_MinP.y - tickLength, bbox.m_MinP.z);
    gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(p1, color, opacity, code),
                                    Gesture::Graphics::VertsCode(p2, color, opacity, code) },
                                  s_lineThickness);

    // Top front edge
    p1 = glm::vec3(x, bbox.m_MaxP.y, bbox.m_MaxP.z);
    p2 = glm::vec3(x, bbox.m_MaxP.y + tickLength, bbox.m_MaxP.z);
    gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(p1, color, opacity, code),
                                    Gesture::Graphics::VertsCode(p2, color, opacity, code) },
                                  s_lineThickness);

    // Top back edge
    p1 = glm::vec3(x, bbox.m_MaxP.y, bbox.m_MinP.z);
    p2 = glm::vec3(x, bbox.m_MaxP.y + tickLength, bbox.m_MinP.z);
    gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(p1, color, opacity, code),
                                    Gesture::Graphics::VertsCode(p2, color, opacity, code) },
                                  s_lineThickness);
  }

  // Draw tick marks along Y axis on the side edges
  for (float t = 0.0f; t <= 1.0f; t += tickSpacing) {
    if (t > 1.0f)
      t = 1.0f;

    float y = center.y + (t - 0.5f) * extent.y;

    // Left front edge
    glm::vec3 p1(bbox.m_MinP.x, y, bbox.m_MaxP.z);
    glm::vec3 p2(bbox.m_MinP.x - tickLength, y, bbox.m_MaxP.z);
    gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(p1, color, opacity, code),
                                    Gesture::Graphics::VertsCode(p2, color, opacity, code) },
                                  s_lineThickness);

    // Left back edge
    p1 = glm::vec3(bbox.m_MinP.x, y, bbox.m_MinP.z);
    p2 = glm::vec3(bbox.m_MinP.x - tickLength, y, bbox.m_MinP.z);
    gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(p1, color, opacity, code),
                                    Gesture::Graphics::VertsCode(p2, color, opacity, code) },
                                  s_lineThickness);

    // Right front edge
    p1 = glm::vec3(bbox.m_MaxP.x, y, bbox.m_MaxP.z);
    p2 = glm::vec3(bbox.m_MaxP.x + tickLength, y, bbox.m_MaxP.z);
    gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(p1, color, opacity, code),
                                    Gesture::Graphics::VertsCode(p2, color, opacity, code) },
                                  s_lineThickness);

    // Right back edge
    p1 = glm::vec3(bbox.m_MaxP.x, y, bbox.m_MinP.z);
    p2 = glm::vec3(bbox.m_MaxP.x + tickLength, y, bbox.m_MinP.z);
    gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(p1, color, opacity, code),
                                    Gesture::Graphics::VertsCode(p2, color, opacity, code) },
                                  s_lineThickness);
  }

  // Draw tick marks along Z axis on the vertical edges
  for (float t = 0.0f; t <= 1.0f; t += tickSpacing) {
    if (t > 1.0f)
      t = 1.0f;

    float z = center.z + (t - 0.5f) * extent.z;

    // Bottom left edge
    glm::vec3 p1(bbox.m_MinP.x, bbox.m_MinP.y, z);
    glm::vec3 p2(bbox.m_MinP.x - tickLength, bbox.m_MinP.y, z);
    gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(p1, color, opacity, code),
                                    Gesture::Graphics::VertsCode(p2, color, opacity, code) },
                                  s_lineThickness);

    // Bottom right edge
    p1 = glm::vec3(bbox.m_MaxP.x, bbox.m_MinP.y, z);
    p2 = glm::vec3(bbox.m_MaxP.x + tickLength, bbox.m_MinP.y, z);
    gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(p1, color, opacity, code),
                                    Gesture::Graphics::VertsCode(p2, color, opacity, code) },
                                  s_lineThickness);

    // Top left edge
    p1 = glm::vec3(bbox.m_MinP.x, bbox.m_MaxP.y, z);
    p2 = glm::vec3(bbox.m_MinP.x - tickLength, bbox.m_MaxP.y, z);
    gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(p1, color, opacity, code),
                                    Gesture::Graphics::VertsCode(p2, color, opacity, code) },
                                  s_lineThickness);

    // Top right edge
    p1 = glm::vec3(bbox.m_MaxP.x, bbox.m_MaxP.y, z);
    p2 = glm::vec3(bbox.m_MaxP.x + tickLength, bbox.m_MaxP.y, z);
    gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(p1, color, opacity, code),
                                    Gesture::Graphics::VertsCode(p2, color, opacity, code) },
                                  s_lineThickness);
  }
}
