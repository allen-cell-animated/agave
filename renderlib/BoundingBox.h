#pragma once

#include "Enumerations.h"

#include "glm.h"

#include <array>
#include <sstream>

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

  void GetCorners(std::array<glm::vec3, 8>& corners) const
  {
    corners[0] = glm::vec3(m_MinP.x, m_MinP.y, m_MinP.z);
    corners[1] = glm::vec3(m_MaxP.x, m_MinP.y, m_MinP.z);
    corners[2] = glm::vec3(m_MaxP.x, m_MaxP.y, m_MinP.z);
    corners[3] = glm::vec3(m_MinP.x, m_MaxP.y, m_MinP.z);
    corners[4] = glm::vec3(m_MinP.x, m_MinP.y, m_MaxP.z);
    corners[5] = glm::vec3(m_MaxP.x, m_MinP.y, m_MaxP.z);
    corners[6] = glm::vec3(m_MaxP.x, m_MaxP.y, m_MaxP.z);
    corners[7] = glm::vec3(m_MinP.x, m_MaxP.y, m_MaxP.z);
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
