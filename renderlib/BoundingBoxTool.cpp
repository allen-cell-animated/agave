#include "BoundingBoxTool.h"

#include "AppScene.h"
#include "BoundingBox.h"
#include "MathUtil.h"

#include <unordered_set>

static const float s_lineThickness = 4.0f;
// Edge represented canonically as {minIndex, maxIndex}
struct Edge
{
  int a, b;
  bool operator==(const Edge& o) const { return a == o.a && b == o.b; }
};

struct EdgeHash
{
  size_t operator()(const Edge& e) const { return (size_t(e.a) << 16) ^ size_t(e.b); }
};

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
  static const int faces[6][4] = {
    { 0, 1, 2, 3 }, // bottom (-z)
    { 4, 5, 6, 7 }, // top (+z)
    { 0, 1, 5, 4 }, // front (-y)
    { 2, 3, 7, 6 }, // back (+y)
    { 0, 3, 7, 4 }, // left (-x)
    { 1, 2, 6, 5 }  // right (+x)
  };
  std::unordered_set<Edge, EdgeHash> edges;

  glm::vec3 dir = scene.camera.m_N; // glm::normalize(camDir);

  for (int f = 0; f < 6; ++f) {
    int i0 = faces[f][0], i1 = faces[f][1], i2 = faces[f][2];
    glm::vec3 v0 = corners[i0], v1 = corners[i1], v2 = corners[i2];

    // Face normal
    glm::vec3 n = glm::normalize(glm::cross(v1 - v0, v2 - v0));

    // Vector from face to camera
    glm::vec3 toCam = scene.camera.m_From - v0;

    // If normal points towards camera, it's front-facing
    if (glm::dot(n, toCam) > 0) {
      // Add all 4 edges of this face
      for (int e = 0; e < 4; ++e) {
        int a = faces[f][e];
        int b = faces[f][(e + 1) % 4];
        Edge edge = { std::min(a, b), std::max(a, b) };
        edges.insert(edge);
      }
    }
  }

  // Get bounding box color from scene material
  glm::vec3 color = glm::vec3(theScene->m_material.m_boundingBoxColor[0],
                              theScene->m_material.m_boundingBoxColor[1],
                              theScene->m_material.m_boundingBoxColor[2]);

  float opacity = 0.5f;
  uint32_t code = Gesture::Graphics::k_noSelectionCode;

  for (auto edge : edges) {
    gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(corners[edge.a], color, opacity, code),
                                    Gesture::Graphics::VertsCode(corners[edge.b], color, opacity, code) },
                                  s_lineThickness);
    if (theScene->m_showScaleBar && scene.camera.m_Projection != ProjectionMode::ORTHOGRAPHIC) {
      drawEdgeTickMarks(corners[edge.a], corners[edge.b], bbox, gesture, color, opacity, code);
    }
  }
#if 0
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
#endif
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

void
BoundingBoxTool::drawEdgeTickMarks(const glm::vec3& vertex1,
                                   const glm::vec3& vertex2,
                                   const CBoundingBox& bbox,
                                   Gesture& gesture,
                                   const glm::vec3& color,
                                   float opacity,
                                   uint32_t code)
{
  glm::vec3 extent = bbox.GetExtent();

  // Length of tick mark lines as a fraction of the smallest dimension
  float minDim = glm::min(glm::min(extent.x, extent.y), extent.z);
  float tickLength = minDim * 0.025f; // 2.5% of smallest dimension

  // Calculate physical scale based on extent - use largest dimension
  float maxDim = glm::max(glm::max(extent.x, extent.y), extent.z);
  float tickSpacing = computePhysicalScaleBarSize(maxDim) / maxDim; // Normalized tick spacing

  // Ensure we don't create too many or too few tick marks
  tickSpacing = glm::max(tickSpacing, 0.1f); // At least 10 ticks max
  tickSpacing = glm::min(tickSpacing, 0.5f); // At most 2 ticks per dimension

  // Calculate edge direction and length
  glm::vec3 edgeVector = vertex2 - vertex1;
  float edgeLength = glm::length(edgeVector);
  glm::vec3 edgeDirection = glm::normalize(edgeVector);

  // Calculate tick direction perpendicular to the edge
  // Choose the best perpendicular direction based on edge orientation
  glm::vec3 tickDirection;

  // Determine which axis the edge is primarily aligned with
  glm::vec3 absEdgeDir = glm::abs(edgeDirection);

  glm::vec3 center = bbox.GetCenter();
  glm::vec3 edgeMidpoint = (vertex1 + vertex2) * 0.5f;
  glm::vec3 toCenter = center - edgeMidpoint;

  if (absEdgeDir.x > absEdgeDir.y && absEdgeDir.x > absEdgeDir.z) {
    // Edge is primarily along X axis
    // Use Y or Z for tick direction, preferring the one that points outward from bbox center
    if (glm::abs(toCenter.y) > glm::abs(toCenter.z)) {
      tickDirection = glm::vec3(0, toCenter.y > 0 ? -1 : 1, 0); // Point away from center
    } else {
      tickDirection = glm::vec3(0, 0, toCenter.z > 0 ? -1 : 1); // Point away from center
    }
  } else if (absEdgeDir.y > absEdgeDir.z) {
    // Edge is primarily along Y axis

    if (glm::abs(toCenter.x) > glm::abs(toCenter.z)) {
      tickDirection = glm::vec3(toCenter.x > 0 ? -1 : 1, 0, 0); // Point away from center
    } else {
      tickDirection = glm::vec3(0, 0, toCenter.z > 0 ? -1 : 1); // Point away from center
    }
  } else {
    // Edge is primarily along Z axis

    if (glm::abs(toCenter.x) > glm::abs(toCenter.y)) {
      tickDirection = glm::vec3(toCenter.x > 0 ? -1 : 1, 0, 0); // Point away from center
    } else {
      tickDirection = glm::vec3(0, toCenter.y > 0 ? -1 : 1, 0); // Point away from center
    }
  }

  // Draw tick marks along the edge
  for (float t = 0.0f; t <= 1.0f; t += tickSpacing) {
    if (t > 1.0f)
      t = 1.0f;

    // Calculate position along the edge
    glm::vec3 edgePoint = vertex1 + t * edgeVector;

    // Calculate tick mark endpoints
    glm::vec3 tickStart = edgePoint;
    glm::vec3 tickEnd = edgePoint + tickDirection * tickLength;

    // Draw the tick mark
    gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(tickStart, color, opacity, code),
                                    Gesture::Graphics::VertsCode(tickEnd, color, opacity, code) },
                                  s_lineThickness);
  }
}
