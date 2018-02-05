#include "Camera2.cuh"

#include "Geometry.h"
#include "helper_math.cuh"
#include "MonteCarlo.cuh"
#include "RNG.cuh"

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

	// Uncomment this line to flip the image to correspond with our other internal viewers that are mirroring the image in some way.
	//ScreenPoint.x = cam.m_Screen[0][1] - (cam.m_InvScreen[0] * Pixel.x);
	ScreenPoint.x = cam.m_Screen[0][0] + (cam.m_InvScreen[0] * Pixel.x);
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
