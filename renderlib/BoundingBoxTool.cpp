#include "BoundingBoxTool.h"

#include "AppScene.h"
#include "BoundingBox.h"
#include "ImageXYZC.h"
#include "Logging.h"
#include "MathUtil.h"

#include <unordered_set>

std::vector<glm::vec3>
computeEdgeTickMarkVertices(const glm::vec3& vertex1,
                            const glm::vec3& vertex2,
                            const CBoundingBox& bbox,
                            float maxNumTickMarks,
                            float tickLength)
{
  glm::vec3 extent = bbox.GetExtent();

  // Calculate edge direction and length
  glm::vec3 edgeVector = vertex2 - vertex1;
  glm::vec3 edgeDirection = glm::normalize(edgeVector);

  // Calculate tick direction perpendicular to the edge
  // Choose the best perpendicular direction based on edge orientation
  glm::vec3 tickDirection;

  // Determine which axis the edge is primarily aligned with
  glm::vec3 absEdgeDir = glm::abs(edgeDirection);

  glm::vec3 center = bbox.GetCenter();
  glm::vec3 edgeMidpoint = (vertex1 + vertex2) * 0.5f;
  glm::vec3 toCenter = center - edgeMidpoint;

  float tickSpacing = 1.0f;
  if (absEdgeDir.x > absEdgeDir.y && absEdgeDir.x > absEdgeDir.z) {
    // Edge is primarily along X axis
    // Use Y or Z for tick direction, preferring the one that points outward from bbox center
    if (glm::abs(toCenter.y) > glm::abs(toCenter.z)) {
      tickDirection = glm::vec3(0, toCenter.y > 0 ? -1 : 1, 0); // Point away from center
    } else {
      tickDirection = glm::vec3(0, 0, toCenter.z > 0 ? -1 : 1); // Point away from center
    }
    tickSpacing = 1.0f / (extent.x * maxNumTickMarks);

  } else if (absEdgeDir.y > absEdgeDir.z) {
    // Edge is primarily along Y axis

    if (glm::abs(toCenter.x) > glm::abs(toCenter.z)) {
      tickDirection = glm::vec3(toCenter.x > 0 ? -1 : 1, 0, 0); // Point away from center
    } else {
      tickDirection = glm::vec3(0, 0, toCenter.z > 0 ? -1 : 1); // Point away from center
    }
    tickSpacing = 1.0f / (extent.y * maxNumTickMarks);
  } else {
    // Edge is primarily along Z axis

    if (glm::abs(toCenter.x) > glm::abs(toCenter.y)) {
      tickDirection = glm::vec3(toCenter.x > 0 ? -1 : 1, 0, 0); // Point away from center
    } else {
      tickDirection = glm::vec3(0, toCenter.y > 0 ? -1 : 1, 0); // Point away from center
    }
    tickSpacing = 1.0f / (extent.z * maxNumTickMarks);
  }

  std::vector<glm::vec3> tickVertices;
  // Draw tick marks along the edge
  for (float t = 0.0f; t <= 1.0f; t += tickSpacing) {
    if (t > 1.0f)
      t = 1.0f;

    // Calculate position along the edge
    // TODO the 1-t here is to match up with the tickmarks in Utils.cpp createTickMarks
    glm::vec3 edgePoint = vertex1 + (1.0f - t) * edgeVector;

    // Calculate tick mark endpoints
    glm::vec3 tickStart = edgePoint;
    glm::vec3 tickEnd = edgePoint + tickDirection * tickLength;
    tickVertices.push_back(tickStart);
    tickVertices.push_back(tickEnd);
  }
  return tickVertices;
}

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

  // Make the edges go in a particular direction so that the tickmarks are lined up on both sides.
  // These edges are set up to go from negative to positive values of the corner coordinates.
  // The indices of the edge are indices into the corners array.
  static const Edge edgesArray[12] = {
    { 0, 1 }, { 1, 2 }, { 3, 2 }, { 0, 3 }, // bottom (-z) face
    { 4, 5 }, { 5, 6 }, { 7, 6 }, { 4, 7 }, // top (+z) face
    { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }  // vertical edges
  };

  // Corner coordinate indices of the 4 vertices of each face, in somewhat arbitrary order.
  static const int faces[6][4] = {
    { 0, 1, 2, 3 }, // bottom (-z)
    { 4, 5, 6, 7 }, // top (+z)
    { 0, 1, 5, 4 }, // front (-y)
    { 2, 3, 7, 6 }, // back (+y)
    { 0, 3, 7, 4 }, // left (-x)
    { 1, 2, 6, 5 }  // right (+x)
  };

  // Face normals for each face
  static const glm::vec3 faceNormals[6] = {
    { 0, 0, -1 }, // bottom (-z)
    { 0, 0, 1 },  // top (+z)
    { 0, -1, 0 }, // front (-y)
    { 0, 1, 0 },  // back (+y)
    { -1, 0, 0 }, // left (-x)
    { 1, 0, 0 }   // right (+x)
  };

  // Each edge belongs to 2 faces.  Set up an array of indices to the 2 face normals for each edge.
  static const int edgeToFace[12][2] = {
    { 0, 2 }, { 0, 5 }, { 0, 3 }, { 0, 4 }, // bottom (-z) face
    { 1, 2 }, { 1, 5 }, { 1, 3 }, { 1, 4 }, // top (+z) face
    { 2, 4 }, { 2, 5 }, { 3, 5 }, { 3, 4 }  // vertical edges
  };

  std::vector<Edge> frontFacingEdges;

  // loop over all edges
  for (int i = 0; i < 12; ++i) {
    // loop over the two faces of each edge:
    for (int j = 0; j < 2; ++j) {
      int faceIndex = edgeToFace[i][j];
      glm::vec3 faceNormal = faceNormals[faceIndex];
      auto edge = edgesArray[i];
      // Vector from face to camera (pick either edge vertex for this, or maybe even the midpoint)
      glm::vec3 toCam = scene.camera.m_From - corners[edge.a];
      // If normal points towards camera, it's front-facing
      if (glm::dot(faceNormal, toCam) > 0) {
        frontFacingEdges.push_back(edgesArray[i]);
        // Do not check the other face; we don't want to add it twice.
        break;
      }
    }
  }

  // Get bounding box color from scene material
  glm::vec3 color = glm::vec3(theScene->m_material.m_boundingBoxColor[0],
                              theScene->m_material.m_boundingBoxColor[1],
                              theScene->m_material.m_boundingBoxColor[2]);

  float opacity = 0.5f;
  uint32_t code = Gesture::Graphics::k_noSelectionCode;

  const glm::vec3 volumePhysicalSize = theScene->m_volume->getPhysicalDimensions();
  float maxPhysicalDim = std::max(volumePhysicalSize.x, std::max(volumePhysicalSize.y, volumePhysicalSize.z));
  const float tickMarkPhysicalLength = computePhysicalScaleBarSize(maxPhysicalDim);
  const float maxNumTickMarks = maxPhysicalDim / tickMarkPhysicalLength;

  for (auto edge : frontFacingEdges) {
    gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(corners[edge.a], color, opacity, code),
                                    Gesture::Graphics::VertsCode(corners[edge.b], color, opacity, code) },
                                  s_lineThickness);
    if (theScene->m_showScaleBar && scene.camera.m_Projection != ProjectionMode::ORTHOGRAPHIC) {
      glm::vec3 extent = bbox.GetExtent();

      // Length of tick mark lines as a fraction of the smallest dimension
      float minDim = glm::min(glm::min(extent.x, extent.y), extent.z);
      float tickLength = minDim * 0.05f; // 5% of smallest dimension

      std::vector<glm::vec3> tickVertices =
        computeEdgeTickMarkVertices(corners[edge.a], corners[edge.b], bbox, maxNumTickMarks, tickLength);
      if (tickVertices.size() >= 2) {
        // loop and add a line strip for each tick mark
        for (size_t i = 0; i + 1 < tickVertices.size(); i += 2) {
          gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(tickVertices[i], color, opacity, code),
                                          Gesture::Graphics::VertsCode(tickVertices[i + 1], color, opacity, code) },
                                        s_lineThickness);
        }
      }
    }
  }
}
