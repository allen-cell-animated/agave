#include "BoundingBoxTool.h"

#include "AppScene.h"
#include "BoundingBox.h"
#include "ImageXYZC.h"
#include "Logging.h"
#include "MathUtil.h"

#include <unordered_set>

static const float s_lineThickness = 4.0f;

struct EdgeHash
{
  size_t operator()(const CBoundingBox::Edge& e) const { return (size_t(e.a) << 16) ^ size_t(e.b); }
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
  std::array<glm::vec3, 8> corners;
  bbox.GetCorners(corners);

  std::vector<CBoundingBox::Edge> frontFacingEdges;
  std::vector<CBoundingBox::Edge> backFacingEdges;

  // loop over all edges
  for (int i = 0; i < 12; ++i) {
    bool isFrontFacing = false;
    auto edge = CBoundingBox::EDGES_ARRAY[i];
    // loop over the two faces of each edge:
    for (int j = 0; j < 2; ++j) {
      int faceIndex = CBoundingBox::EDGE_TO_FACE[i][j];
      glm::vec3 faceNormal = CBoundingBox::FACE_NORMALS[faceIndex];
      // Vector from face to camera (pick either edge vertex for this, or maybe even the midpoint)
      glm::vec3 toCam = scene.camera.m_From - corners[edge.a];
      // If normal points towards camera, it's front-facing
      if (glm::dot(faceNormal, toCam) > 0) {
        frontFacingEdges.push_back(edge);
        isFrontFacing = true;
        // Do not check the other face; we don't want to add it twice.
        break;
      }
    }
    if (!isFrontFacing) {
      backFacingEdges.push_back(edge);
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
  const float tickMarkPhysicalSpacing = computePhysicalScaleBarSize(maxPhysicalDim);
  const float maxNumTickMarks = maxPhysicalDim / tickMarkPhysicalSpacing;

  for (auto edge : frontFacingEdges) {
    gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(corners[edge.a], color, opacity, code),
                                    Gesture::Graphics::VertsCode(corners[edge.b], color, opacity, code) },
                                  s_lineThickness);
    if (theScene->m_showScaleBar && scene.camera.m_Projection != ProjectionMode::ORTHOGRAPHIC) {
      glm::vec3 extent = bbox.GetExtent();

      // Length of tick mark lines as a fraction of the smallest dimension
      float minDim = glm::min(glm::min(extent.x, extent.y), extent.z);
      float tickLength = minDim * 0.025f; // 5% of smallest dimension

      std::vector<glm::vec3> tickVertices;
      bbox.GetEdgeTickMarkVertices(corners[edge.a], corners[edge.b], maxNumTickMarks, tickLength, tickVertices);
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
  for (auto edge : backFacingEdges) {
    gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(corners[edge.a], color, opacity, code),
                                    Gesture::Graphics::VertsCode(corners[edge.b], color, opacity, code) },
                                  s_lineThickness,
                                  false,
                                  Gesture::Graphics::CommandSequence::k3dStackedUnderlay);
    if (theScene->m_showScaleBar && scene.camera.m_Projection != ProjectionMode::ORTHOGRAPHIC) {
      glm::vec3 extent = bbox.GetExtent();

      // Length of tick mark lines as a fraction of the smallest dimension
      float minDim = glm::min(glm::min(extent.x, extent.y), extent.z);
      float tickLength = minDim * 0.025f; // 5% of smallest dimension

      std::vector<glm::vec3> tickVertices;
      bbox.GetEdgeTickMarkVertices(corners[edge.a], corners[edge.b], maxNumTickMarks, tickLength, tickVertices);
      if (tickVertices.size() >= 2) {
        // loop and add a line strip for each tick mark
        for (size_t i = 0; i + 1 < tickVertices.size(); i += 2) {
          gesture.graphics.addLineStrip({ Gesture::Graphics::VertsCode(tickVertices[i], color, opacity, code),
                                          Gesture::Graphics::VertsCode(tickVertices[i + 1], color, opacity, code) },
                                        s_lineThickness,
                                        false,
                                        Gesture::Graphics::CommandSequence::k3dStackedUnderlay);
        }
      }
    }
  }
}
