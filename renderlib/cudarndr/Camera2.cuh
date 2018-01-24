#pragma once

#include "Geometry.h"
#include "helper_math.cuh"
#include "MonteCarlo.cuh"
#include "RNG.cuh"

// enough data to generate a camera ray
// float3 is used to have a struct that has no ctor, so that this can be stored in __constant__ memory.
struct CudaCamera {
	float3 m_From;
	float3 m_N;
	float3 m_U;
	float3 m_V;
	float m_ApertureSize;
	float m_FocalDistance;
	float m_InvScreen[2];
	float m_Screen[2][2];
};

DEV Vec3f toVec3(const float3& f)
{
	return Vec3f(f.x, f.y, f.z);
}
DEV Vec3f Normalize(const float3& v)
{
	return Normalize(Vec3f(v.x, v.y, v.z));
}

DEV void GenerateRay(const CudaCamera& cam, const Vec2f& Pixel, const Vec2f& ApertureRnd, Vec3f& RayO, Vec3f& RayD)
{
	Vec2f ScreenPoint;

	ScreenPoint.x = cam.m_Screen[0][1] - (cam.m_InvScreen[0] * Pixel.x);
	ScreenPoint.y = cam.m_Screen[1][0] + (cam.m_InvScreen[1] * Pixel.y);

	RayO = toVec3(cam.m_From);
	RayD = Normalize(cam.m_N + (-ScreenPoint.x * cam.m_U) + (-ScreenPoint.y * cam.m_V));

	if (cam.m_ApertureSize != 0.0f)
	{
		Vec2f LensUV = cam.m_ApertureSize * ConcentricSampleDisk(ApertureRnd);

		Vec3f LI = toVec3(cam.m_U * LensUV.x + cam.m_V * LensUV.y);
		RayO += LI;
		RayD = Normalize((RayD * cam.m_FocalDistance) - LI);
	}
}

