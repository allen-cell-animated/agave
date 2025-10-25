#include "BoundingBox.h"
#include "glm.h"

// Make the edges go in a particular direction so that the tickmarks are lined up on both sides.
// These edges are set up to go from negative to positive values of the corner coordinates.
// The indices of the edge are indices into the corners array.
const CBoundingBox::Edge CBoundingBox::EDGES_ARRAY[CBoundingBox::NUM_EDGES] = {
  { 0, 1 }, { 1, 2 }, { 3, 2 }, { 0, 3 }, // bottom (-z) face
  { 4, 5 }, { 5, 6 }, { 7, 6 }, { 4, 7 }, // top (+z) face
  { 0, 4 }, { 1, 5 }, { 2, 6 }, { 3, 7 }  // vertical edges
};

// what is the axis aligned with each edge
const glm::vec3 CBoundingBox::EDGE_DIRECTION[CBoundingBox::NUM_EDGES] = {
  { 1, 0, 0 }, { 0, 1, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, // bottom (-z) face
  { 1, 0, 0 }, { 0, 1, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, // top (+z) face
  { 0, 0, 1 }, { 0, 0, 1 }, { 0, 0, 1 }, { 0, 0, 1 }  // vertical edges
};
const EAxis CBoundingBox::EDGE_AXIS[CBoundingBox::NUM_EDGES] = {
  AxisX, AxisY, AxisX, AxisY, // bottom (-z) face
  AxisX, AxisY, AxisX, AxisY, // top (+z) face
  AxisZ, AxisZ, AxisZ, AxisZ  // vertical edges
};

// Corner coordinate indices of the 4 vertices of each face, in somewhat arbitrary order.
const int CBoundingBox::FACES[CBoundingBox::NUM_FACES][4] = {
  { 0, 1, 2, 3 }, // bottom (-z)
  { 4, 5, 6, 7 }, // top (+z)
  { 0, 1, 5, 4 }, // front (-y)
  { 2, 3, 7, 6 }, // back (+y)
  { 0, 3, 7, 4 }, // left (-x)
  { 1, 2, 6, 5 }  // right (+x)
};

// Face normals for each face.. assumes axis-aligned.
const glm::vec3 CBoundingBox::FACE_NORMALS[CBoundingBox::NUM_FACES] = {
  { 0, 0, -1 }, // bottom (-z)
  { 0, 0, 1 },  // top (+z)
  { 0, -1, 0 }, // front (-y)
  { 0, 1, 0 },  // back (+y)
  { -1, 0, 0 }, // left (-x)
  { 1, 0, 0 }   // right (+x)
};

// Each edge belongs to 2 faces.  Set up an array of indices to the 2 faces for each edge.
const int CBoundingBox::EDGE_TO_FACE[CBoundingBox::NUM_EDGES][2] = {
  { 0, 2 }, { 0, 5 }, { 0, 3 }, { 0, 4 }, // bottom (-z) face
  { 1, 2 }, { 1, 5 }, { 1, 3 }, { 1, 4 }, // top (+z) face
  { 2, 4 }, { 2, 5 }, { 3, 5 }, { 3, 4 }  // vertical edges
};

// pass in two vertices of the edge (should be corners from GetCorners)
void
CBoundingBox::GetEdgeTickMarkVertices(const glm::vec3& vertex1,
                                      const glm::vec3& vertex2,
                                      float maxNumTickMarks,
                                      float tickLength,
                                      std::vector<glm::vec3>& tickVertices) const
{
  glm::vec3 extent = GetExtent();

  // Calculate edge direction and length
  glm::vec3 edgeVector = vertex2 - vertex1;
  glm::vec3 edgeDirection = glm::normalize(edgeVector);

  // Calculate tick direction perpendicular to the edge
  // Choose the best perpendicular direction based on edge orientation
  glm::vec3 tickDirection;

  // Determine which axis the edge is primarily aligned with
  glm::vec3 absEdgeDir = glm::abs(edgeDirection);

  glm::vec3 center = GetCenter();
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
}
