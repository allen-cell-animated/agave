#pragma once

#include "Enumerations.h"

#include "glm.h"

#include <array>
#include <sstream>
#include <vector>

#define NUM_BBOX_CORNERS 8

class CBoundingBox
{
public:
  glm::vec3 m_MinP;
  glm::vec3 m_MaxP;

  CBoundingBox(void)
    : m_MinP(FLT_MAX, FLT_MAX, FLT_MAX)
    , m_MaxP(-FLT_MAX, -FLT_MAX, -FLT_MAX)
  {
  }

  CBoundingBox(const glm::vec3& v1, const glm::vec3& v2)
    : m_MinP(v1)
    , m_MaxP(v2)
  {
  }

  CBoundingBox& operator=(const CBoundingBox& B)
  {
    m_MinP = B.m_MinP;
    m_MaxP = B.m_MaxP;

    return *this;
  }

  // Adds a point to this bounding box
  CBoundingBox& operator+=(const glm::vec3& P)
  {
    if (!Contains(P)) {
      for (int i = 0; i < 3; i++) {
        if (P[i] < m_MinP[i])
          m_MinP[i] = P[i];

        if (P[i] > m_MaxP[i])
          m_MaxP[i] = P[i];
      }
    }

    return *this;
  }

  // Adds a bounding box to this bounding box
  CBoundingBox& operator+=(const CBoundingBox& B)
  {
    *this += B.m_MinP;
    *this += B.m_MaxP;

    return *this;
  }

  glm::vec3& operator[](int i) { return (&m_MinP)[i]; }

  const glm::vec3& operator[](int i) const { return (&m_MinP)[i]; }

  float LengthX(void) const { return fabs(m_MaxP.x - m_MinP.x); };
  float LengthY(void) const { return fabs(m_MaxP.y - m_MinP.y); };
  float LengthZ(void) const { return fabs(m_MaxP.z - m_MinP.z); };

  glm::vec3 GetCenter(void) const
  {
    return glm::vec3(0.5f * (m_MinP.x + m_MaxP.x), 0.5f * (m_MinP.y + m_MaxP.y), 0.5f * (m_MinP.z + m_MaxP.z));
  }

  EContainment Contains(const glm::vec3& P) const
  {
    for (int i = 0; i < 3; i++) {
      if (P[i] < m_MinP[i] || P[i] > m_MaxP[i])
        return ContainmentNone;
    }

    return ContainmentFull;
  };

  EContainment Contains(const glm::vec3* pPoints, long PointCount) const
  {
    long Contain = 0;

    for (int i = 0; i < PointCount; i++) {
      if (Contains(pPoints[i]) == ContainmentFull)
        Contain++;
    }

    if (Contain == 0)
      return ContainmentNone;
    else {
      if (Contain == PointCount)
        return ContainmentFull;
      else
        return ContainmentPartial;
    }
  }

  EContainment Contains(const CBoundingBox& B) const
  {
    bool ContainsMin = false, ContainsMax = false;

    if (Contains(B.m_MinP) == ContainmentFull)
      ContainsMin = true;

    if (Contains(B.m_MaxP) == ContainmentFull)
      ContainsMax = true;

    if (!ContainsMin && !ContainsMax)
      return ContainmentNone;
    else {
      if (ContainsMin && ContainsMax)
        return ContainmentFull;
      else
        return ContainmentPartial;
    }
  }

  EAxis GetDominantAxis(void) const
  {
    return (LengthX() > LengthY() && LengthX() > LengthZ()) ? AxisX : ((LengthY() > LengthZ()) ? AxisY : AxisZ);
  }

  glm::vec3 GetMinP(void) const { return m_MinP; }
  glm::vec3 GetInvMinP(void) const { return glm::vec3(1.0f) / m_MinP; }
  void SetMinP(glm::vec3 MinP) { m_MinP = MinP; }
  glm::vec3 GetMaxP(void) const { return m_MaxP; }
  glm::vec3 GetInvMaxP(void) const { return glm::vec3(1.0f) / m_MaxP; }
  void SetMaxP(glm::vec3 MaxP) { m_MaxP = MaxP; }

  float GetMaxLength(EAxis* pAxis = NULL) const
  {
    if (pAxis)
      *pAxis = GetDominantAxis();

    const glm::vec3& MinMax = GetExtent();

    return MinMax[GetDominantAxis()];
  }

  float GetDiagonalLength() const
  {
    return sqrt(LengthX() * LengthX() + LengthY() * LengthY() + LengthZ() * LengthZ());
  }

  float HalfSurfaceArea(void) const
  {
    const glm::vec3 e(GetExtent());
    return e.x * e.y + e.y * e.z + e.x * e.z;
  }

  float GetArea(void) const
  {
    const glm::vec3 ext(m_MaxP - m_MinP);
    return float(ext.x) * float(ext.y) + float(ext.y) * float(ext.z) + float(ext.x) * float(ext.z);
  }

  glm::vec3 GetExtent(void) const { return m_MaxP - m_MinP; }
  glm::vec3 GetInverseExtent(void) const
  {
    glm::vec3 v = GetExtent();
    return glm::vec3(1.0f / v.x, 1.0f / v.y, 1.0f / v.z);
  }

  float GetEquivalentRadius(void) const { return 0.5f * glm::length(GetExtent()); }

  bool Inside(const glm::vec3& pt)
  {
    return (pt.x >= m_MinP.x && pt.x <= m_MaxP.x && pt.y >= m_MinP.y && pt.y <= m_MaxP.y && pt.z >= m_MinP.z &&
            pt.z <= m_MaxP.z);
  }

  std::string ToString() const
  {
    std::stringstream ss;
    ss << "Min(" << m_MinP.x << "," << m_MinP.y << "," << m_MinP.z << "), Max(" << m_MaxP.x << "," << m_MaxP.y << ","
       << m_MaxP.z << ")";
    return ss.str();
  }

  void Extend(float f)
  {
    m_MinP -= glm::vec3(f);
    m_MaxP += glm::vec3(f);
  }

  void GetCorners(std::array<glm::vec3, NUM_BBOX_CORNERS>& corners) const
  {
    corners[0] = glm::vec3(m_MinP.x, m_MinP.y, m_MinP.z); // min corner
    corners[1] = glm::vec3(m_MaxP.x, m_MinP.y, m_MinP.z);
    corners[2] = glm::vec3(m_MaxP.x, m_MaxP.y, m_MinP.z);
    corners[3] = glm::vec3(m_MinP.x, m_MaxP.y, m_MinP.z);
    corners[4] = glm::vec3(m_MinP.x, m_MinP.y, m_MaxP.z);
    corners[5] = glm::vec3(m_MaxP.x, m_MinP.y, m_MaxP.z);
    corners[6] = glm::vec3(m_MaxP.x, m_MaxP.y, m_MaxP.z); // max corner
    corners[7] = glm::vec3(m_MinP.x, m_MaxP.y, m_MaxP.z);
  }

  // pass in two vertices of the edge (should be corners from GetCorners)
  void GetEdgeTickMarkVertices(const glm::vec3& vertex1,
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

#if 0
	// Performs a line box intersection
	bool Intersect(CRay& R, float* pMinT = NULL, float* pMaxT = NULL)
	{
		// Compute intersection of line with all six bounding box planes
		const glm::vec3 InvR = glm::vec3(1.0f / R.m_D.x, 1.0f / R.m_D.y, 1.0f / R.m_D.z);
		const glm::vec3 BotT = InvR * (m_MinP - glm::vec3(R.m_O.x, R.m_O.y, R.m_O.z));
		const glm::vec3 TopT = InvR * (m_MaxP - glm::vec3(R.m_O.x, R.m_O.y, R.m_O.z));

		// re-order intersections to find smallest and largest on each axis
		const glm::vec3 MinT = glm::vec3(min(TopT.x, BotT.x), min(TopT.y, BotT.y), min(TopT.z, BotT.z));
		const glm::vec3 MaxT = glm::vec3(max(TopT.x, BotT.x), max(TopT.y, BotT.y), max(TopT.z, BotT.z));

		// find the largest tmin and the smallest tmax
		const float LargestMinT		= max(max(MinT.x, MinT.y), max(MinT.x, MinT.z));
		const float SmallestMaxT	= min(min(MaxT.x, MaxT.y), min(MaxT.x, MaxT.z));

		if (pMinT)
			*pMinT = LargestMinT;

		if (pMaxT)
			*pMaxT = SmallestMaxT;

		return SmallestMaxT > LargestMinT;
	}

	bool IntersectP(const CRay& ray, float* hitt0 = NULL, float* hitt1 = NULL)
	{
		float t0 = ray.m_MinT, t1 = ray.m_MaxT;
		
		for (int i = 0; i < 3; ++i)
		{
			// Update interval for _i_th bounding box slab
			float invRayDir = 1.f / ray.m_D[i];
			float tNear = (m_MinP[i] - ray.m_O[i]) * invRayDir;
			float tFar  = (m_MaxP[i] - ray.m_O[i]) * invRayDir;

			// Update parametric interval from slab intersection $t$s
			if (tNear > tFar)
				swap(tNear, tFar);

			t0 = tNear > t0 ? tNear : t0;
			t1 = tFar  < t1 ? tFar  : t1;

			if (t0 > t1)
				return false;
		}

		if (hitt0)
			*hitt0 = t0;

		if (hitt1)
			*hitt1 = t1;

		return true;
	}
#endif
};
