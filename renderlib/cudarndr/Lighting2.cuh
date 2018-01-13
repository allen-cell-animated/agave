#pragma once

#include "helper_math.cuh"

struct CudaLight {
	float			m_Theta;
	float			m_Phi;
	float			m_Width;
	float			m_InvWidth;
	float			m_HalfWidth;
	float			m_InvHalfWidth;
	float			m_Height;
	float			m_InvHeight;
	float			m_HalfHeight;
	float			m_InvHalfHeight;
	float			m_Distance;
	float			m_SkyRadius;
	float3			m_P;
	float3			m_Target;
	float3			m_N;
	float3			m_U;
	float3			m_V;
	float			m_Area;
	float			m_AreaPdf;
	float3	m_Color;
	float3	m_ColorTop;
	float3	m_ColorMiddle;
	float3	m_ColorBottom;
	int				m_T;


	// Samples the light
	DEV CColorXyz SampleL(const Vec3f& P, CRay& Rl, float& Pdf, CLightingSample& LS) const;
	// Intersect ray with light
	DEV bool Intersect(CRay& R, float& T, CColorXyz& L, Vec2f* pUV = NULL, float* pPdf = NULL) const;
	DEV float Pdf(const Vec3f& P, const Vec3f& Wi) const;
	DEV CColorXyz Le(const Vec2f& UV) const;
};
#define MAX_NO_LIGHTS 4
struct CudaLighting {
	int m_NoLights;
	CudaLight m_Lights[MAX_NO_LIGHTS];
};

#if 0
DEV Vec3f toVec3(const float3& f)
{
	return Vec3f(f.x, f.y, f.z);
}
DEV Vec3f Normalize(const float3& v)
{
	return Normalize(Vec3f(v.x, v.y, v.z));
}
#endif

DEV CColorXyz ToXYZ(const float3& f) {
	return CColorXyz::FromRGB(f.x, f.y, f.z);
}
DEV CColorRgbHdr toRGB(const float3& f) {
	return CColorRgbHdr(f.x, f.y, f.z);
}

// Samples the light
DEV CColorXyz CudaLight::SampleL(const Vec3f& P, CRay& Rl, float& Pdf, CLightingSample& LS) const
{
	CColorXyz L = SPEC_BLACK;

	if (m_T == 0)
	{
		Rl.m_O = toVec3(m_P + ((-0.5f + LS.m_LightSample.m_Pos.x) * m_Width * m_U) + ((-0.5f + LS.m_LightSample.m_Pos.y) * m_Height * m_V));
		Rl.m_D = Normalize(P - Rl.m_O);
		L = Dot(Rl.m_D, toVec3(m_N)) > 0.0f ? Le(Vec2f(0.0f)) : SPEC_BLACK;
		Pdf = AbsDot(Rl.m_D, toVec3(m_N)) > 0.0f ? DistanceSquared(P, Rl.m_O) / (AbsDot(Rl.m_D, toVec3(m_N)) * m_Area) : 0.0f;
	}

	if (m_T == 1)
	{
		Rl.m_O = toVec3(m_P) + m_SkyRadius * UniformSampleSphere(LS.m_LightSample.m_Pos);
		Rl.m_D = Normalize(P - Rl.m_O);
		L = Le(Vec2f(1.0f) - 2.0f * LS.m_LightSample.m_Pos);
		Pdf = powf(m_SkyRadius, 2.0f) / m_Area;
	}

	Rl.m_MinT = 0.0f;
	Rl.m_MaxT = (P - Rl.m_O).Length();

	return L;
}

// Intersect ray with light
DEV bool CudaLight::Intersect(CRay& R, float& T, CColorXyz& L, Vec2f* pUV, float* pPdf) const
{
	if (m_T == 0)
	{
		// Compute projection
		const float DotN = Dot(R.m_D, toVec3(m_N));

		// Rays is co-planar with light surface
		if (DotN >= 0.0f)
			return false;

		// Compute hit distance
		T = (-m_Distance - Dot(R.m_O, toVec3(m_N))) / DotN;

		// Intersection is in ray's negative direction
		if (T < R.m_MinT || T > R.m_MaxT)
			return false;

		// Determine position on light
		const Vec3f Pl = R(T);

		// Vector from point on area light to center of area light
		const Vec3f Wl = Pl - toVec3(m_P);

		// Compute texture coordinates
		const Vec2f UV = Vec2f(Dot(Wl, toVec3(m_U)), Dot(Wl, toVec3(m_V)));

		// Check if within bounds of light surface
		if (UV.x > m_HalfWidth || UV.x < -m_HalfWidth || UV.y > m_HalfHeight || UV.y < -m_HalfHeight)
			return false;

		R.m_MaxT = T;

		if (pUV)
			*pUV = UV;

		if (DotN < 0.0f)
			L = ToXYZ(m_Color) / m_Area;
		else
			L = SPEC_BLACK;

		if (pPdf)
			*pPdf = DistanceSquared(R.m_O, Pl) / (DotN * m_Area);

		return true;
	}

	if (m_T == 1)
	{
		T = m_SkyRadius;

		// Intersection is in ray's negative direction
		if (T < R.m_MinT || T > R.m_MaxT)
			return false;

		R.m_MaxT = T;

		Vec2f UV = Vec2f(SphericalPhi(R.m_D) * INV_TWO_PI_F, SphericalTheta(R.m_D) * INV_PI_F);

		L = Le(Vec2f(1.0f) - 2.0f * UV);

		if (pPdf)
			*pPdf = powf(m_SkyRadius, 2.0f) / m_Area;

		return true;
	}

	return false;
}

DEV float CudaLight::Pdf(const Vec3f& P, const Vec3f& Wi) const
{
	CColorXyz L;
	Vec2f UV;
	float Pdf = 1.0f;

	CRay Rl = CRay(P, Wi, 0.0f, INF_MAX);

	if (m_T == 0)
	{
		float T = 0.0f;

		if (!Intersect(Rl, T, L, NULL, &Pdf))
			return 0.0f;

		return powf(T, 2.0f) / (AbsDot(toVec3(m_N), -Wi) * m_Area);
	}

	if (m_T == 1)
	{
		return powf(m_SkyRadius, 2.0f) / m_Area;
	}

	return 0.0f;
}

DEV CColorXyz CudaLight::Le(const Vec2f& UV) const
{
	if (m_T == 0)
		return CColorXyz::FromRGB(m_Color.x, m_Color.y, m_Color.z) / m_Area;

	if (m_T == 1)
	{
		if (UV.y > 0.0f)
			return Lerp(fabs(UV.y), toRGB(m_ColorMiddle), toRGB(m_ColorTop)).ToXYZ();
		else
			return Lerp(fabs(UV.y), toRGB(m_ColorMiddle), toRGB(m_ColorBottom)).ToXYZ();
	}

	return SPEC_BLACK;
}
