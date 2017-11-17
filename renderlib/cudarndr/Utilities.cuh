#pragma once

#include <cuda_runtime.h>

DEV inline Vec3f ToVec3f(const float3& V)
{
	return Vec3f(V.x, V.y, V.z);
}

DEV float GetNormalizedIntensity(const Vec3f& P, cudaTextureObject_t texDensity)
{
	const float Intensity = ((float)SHRT_MAX * tex3D<float>(texDensity, P.x * gInvAaBbMax.x, P.y * gInvAaBbMax.y, P.z * gInvAaBbMax.z));

	return (Intensity - gIntensityMin) * gIntensityInvRange;
}


DEV float GetOpacity(const float& NormalizedIntensity)
{
	return NormalizedIntensity;
	//return tex1D(gTexOpacity, NormalizedIntensity);
}

DEV CColorRgbHdr GetDiffuse(const float& NormalizedIntensity)
{
	//float4 Diffuse = tex1D(gTexDiffuse, NormalizedIntensity);
//	float4 Diffuse = make_float4(NormalizedIntensity, NormalizedIntensity, NormalizedIntensity, 1.0);
//	return CColorRgbHdr(Diffuse.x, Diffuse.y, Diffuse.z);
//	return CColorRgbHdr(1.0, 1.0, 1.0);
	return CColorRgbHdr(gDiffuseColor.x, gDiffuseColor.y, gDiffuseColor.z);
}


DEV CColorRgbHdr GetSpecular(const float& NormalizedIntensity)
{
	//float4 Specular = make_float4(NormalizedIntensity, NormalizedIntensity, NormalizedIntensity, 1.0);
	//float4 Specular = tex1D(gTexSpecular, NormalizedIntensity);
	//return CColorRgbHdr(Specular.x, Specular.y, Specular.z);

	return CColorRgbHdr(gSpecularColor.x, gSpecularColor.y, gSpecularColor.z);
}

DEV float GetRoughness(const float& NormalizedIntensity)
{
	return NormalizedIntensity;
	//return tex1D(gTexRoughness, NormalizedIntensity);
}

DEV CColorRgbHdr GetEmission(const float& NormalizedIntensity)
{
	//float4 Emission = make_float4(NormalizedIntensity, NormalizedIntensity, NormalizedIntensity, 1.0);
	//float4 Emission = tex1D(gTexEmission, NormalizedIntensity);
	//return CColorRgbHdr(Emission.x, Emission.y, Emission.z);

	return CColorRgbHdr(gEmissiveColor.x, gEmissiveColor.y, gEmissiveColor.z);
}

DEV inline Vec3f NormalizedGradient(const Vec3f& P, cudaTextureObject_t texDensity)
{
	Vec3f Gradient;

	Gradient.x = (GetNormalizedIntensity(P + ToVec3f(gGradientDeltaX), texDensity) - GetNormalizedIntensity(P - ToVec3f(gGradientDeltaX), texDensity)) * gInvGradientDelta;
	Gradient.y = (GetNormalizedIntensity(P + ToVec3f(gGradientDeltaY), texDensity) - GetNormalizedIntensity(P - ToVec3f(gGradientDeltaY), texDensity)) * gInvGradientDelta;
	Gradient.z = (GetNormalizedIntensity(P + ToVec3f(gGradientDeltaZ), texDensity) - GetNormalizedIntensity(P - ToVec3f(gGradientDeltaZ), texDensity)) * gInvGradientDelta;

	return Normalize(Gradient);
}

DEV float GradientMagnitude(const Vec3f& P, cudaTextureObject_t texGradientMagnitude)
{
	return ((float)SHRT_MAX * tex3D<float>(texGradientMagnitude, P.x * gInvAaBbMax.x, P.y * gInvAaBbMax.y, P.z * gInvAaBbMax.z));
}

DEV bool NearestLight(CScene* pScene, CRay R, CColorXyz& LightColor, Vec3f& Pl, CLight*& pLight, float* pPdf = NULL)
{
	bool Hit = false;
	
	float T = 0.0f;

	CRay RayCopy = R;

	float Pdf = 0.0f;

	for (int i = 0; i < pScene->m_Lighting.m_NoLights; i++)
	{
		if (pScene->m_Lighting.m_Lights[i].Intersect(RayCopy, T, LightColor, NULL, &Pdf))
		{
			Pl		= R(T);
			pLight	= &pScene->m_Lighting.m_Lights[i];
			Hit		= true;
		}
	}
	
	if (pPdf)
		*pPdf = Pdf;

	return Hit;
}

DEV bool IntersectBox(const CRay& R, float* pNearT, float* pFarT)
{
	const Vec3f InvR		= Vec3f(1.0f, 1.0f, 1.0f) / R.m_D;
	const Vec3f BottomT		= InvR * (Vec3f(gAaBbMin.x, gAaBbMin.y, gAaBbMin.z) - R.m_O);
	const Vec3f TopT		= InvR * (Vec3f(gAaBbMax.x, gAaBbMax.y, gAaBbMax.z) - R.m_O);
	const Vec3f MinT		= MinVec3f(TopT, BottomT);
	const Vec3f MaxT		= MaxVec3f(TopT, BottomT);
	const float LargestMinT = fmaxf(fmaxf(MinT.x, MinT.y), fmaxf(MinT.x, MinT.z));
	const float LargestMaxT = fminf(fminf(MaxT.x, MaxT.y), fminf(MaxT.x, MaxT.z));

	*pNearT = LargestMinT;
	*pFarT	= LargestMaxT;

	return LargestMaxT > LargestMinT;
}

DEV CColorXyza CumulativeMovingAverage(const CColorXyza& A, const CColorXyza& Ax, const int& N)
{
//	if (gNoIterations == 0)
//		return CColorXyza(0.0f);

	 return A + ((Ax - A) / max((float)N, 1.0f));
}