#include "Camera2.cuh"

#include "Geometry.h"
#include "helper_math.cuh"
#include "MonteCarlo.cuh"
#include "RNG.cuh"

DEV void GenerateRay(const CudaCamera& cam, const Vec2f& Pixel, const Vec2f& ApertureRnd, float3& RayO, float3& RayD)
{
	Vec2f ScreenPoint;

	ScreenPoint.x = cam.m_Screen[0][0] + (cam.m_InvScreen[0] * Pixel.x);
	ScreenPoint.y = cam.m_Screen[1][0] + (cam.m_InvScreen[1] * Pixel.y);

	RayO = (cam.m_From);
	// negating ScreenPoint.y flips the up/down direction. depends on whether you want pixel 0 at top or bottom
	RayD = normalize(cam.m_N + (ScreenPoint.x * cam.m_U) + (ScreenPoint.y * cam.m_V));

	if (cam.m_ApertureSize != 0.0f)
	{
		Vec2f LensUV = cam.m_ApertureSize * ConcentricSampleDisk(ApertureRnd);

		float3 LI = (cam.m_U * LensUV.x + cam.m_V * LensUV.y);
		RayO += LI;
		RayD = normalize((RayD * cam.m_FocalDistance) - LI);
	}
}
