#pragma once

#include <cuda_runtime.h>

static DEV Vec3f operator + (const Vec3f& a, const float3& b) { return Vec3f(a.x + b.x, a.y + b.y, a.z + b.z); };
static DEV Vec3f operator - (const Vec3f& a, const float3& b) { return Vec3f(a.x - b.x, a.y - b.y, a.z - b.z); };

// this gives the ability to read a float4 and then loop over its elements.
typedef union {
	float4 vec;
	float a[4];
} f4;

DEV float GetNormalizedIntensityMax4ch(const Vec3f& P, const cudaVolume& volumeData, int& ch)
{
	//float factor = (tex3D<float>(volumeData.volumeTexture[5], P.x * gInvAaBbMax.x, P.y * gInvAaBbMax.y, P.z * gInvAaBbMax.z));
	//factor = (factor > 0) ? 1.0 : 0.0;
	f4 intensity;
	intensity.vec = ((float)SHRT_MAX * tex3D<float4>(volumeData.volumeTexture[0], P.x * gInvAaBbMax.x, P.y * gInvAaBbMax.y, P.z * gInvAaBbMax.z));
	float maxIn = 0;
	for (int i = 0; i < min(volumeData.nChannels, 4); ++i) {
		// 0..1
		intensity.a[i] = intensity.a[i] / volumeData.intensityMax[i];
		// transform through LUT
		intensity.a[i] = tex1D<float>(volumeData.lutTexture[i], intensity.a[i]);
		if (intensity.a[i] > maxIn) {
			maxIn = intensity.a[i];
			ch = i;
		}
	}
	return maxIn; // *factor;
}

DEV float GetNormalizedIntensity4ch(const Vec3f& P, const cudaVolume& volumeData, int ch)
{
	float4 Intensity4 = ((float)SHRT_MAX * tex3D<float4>(volumeData.volumeTexture[0], P.x * gInvAaBbMax.x, P.y * gInvAaBbMax.y, P.z * gInvAaBbMax.z));
	// select channel
	float intensity = ch == 0 ? Intensity4.x : ch == 1 ? Intensity4.y : ch == 2 ? Intensity4.z : Intensity4.w;
	intensity = intensity / volumeData.intensityMax[ch];
	//intensity = tex1D<float>(volumeData.lutTexture[ch], intensity);
	return intensity;
}


DEV float GetOpacity(const float& NormalizedIntensity)
{
	// apply lut
	float Intensity = NormalizedIntensity;
	//float Intensity = tex1D<float>(texLut, NormalizedIntensity);
	return Intensity;
}

DEV CColorRgbHdr GetDiffuseN(const float& NormalizedIntensity, const cudaVolume& volumeData, int ch)
{
	//float4 Diffuse = tex1D(gTexDiffuse, NormalizedIntensity);
	//	float4 Diffuse = make_float4(NormalizedIntensity, NormalizedIntensity, NormalizedIntensity, 1.0);
	//	return CColorRgbHdr(Diffuse.x, Diffuse.y, Diffuse.z);
	//	return CColorRgbHdr(1.0, 1.0, 1.0);
	return CColorRgbHdr(volumeData.diffuse[ch*3+0], volumeData.diffuse[ch * 3 + 1], volumeData.diffuse[ch * 3 + 2]);
}

DEV CColorRgbHdr GetSpecularN(const float& NormalizedIntensity, const cudaVolume& volumeData, int ch)
{
	//float4 Diffuse = tex1D(gTexDiffuse, NormalizedIntensity);
	//	float4 Diffuse = make_float4(NormalizedIntensity, NormalizedIntensity, NormalizedIntensity, 1.0);
	//	return CColorRgbHdr(Diffuse.x, Diffuse.y, Diffuse.z);
	//	return CColorRgbHdr(1.0, 1.0, 1.0);
	return CColorRgbHdr(volumeData.specular[ch * 3 + 0], volumeData.specular[ch * 3 + 1], volumeData.specular[ch * 3 + 2]);
}

DEV float GetRoughnessN(const float& NormalizedIntensity, const cudaVolume& volumeData, int ch)
{
	return volumeData.roughness[ch];
	//return NormalizedIntensity;
	//return tex1D(gTexRoughness, NormalizedIntensity);
}

DEV CColorRgbHdr GetEmissionN(const float& NormalizedIntensity, const cudaVolume& volumeData, int ch)
{
	//float4 Diffuse = tex1D(gTexDiffuse, NormalizedIntensity);
	//	float4 Diffuse = make_float4(NormalizedIntensity, NormalizedIntensity, NormalizedIntensity, 1.0);
	//	return CColorRgbHdr(Diffuse.x, Diffuse.y, Diffuse.z);
	//	return CColorRgbHdr(1.0, 1.0, 1.0);
	return CColorRgbHdr(volumeData.emissive[ch * 3 + 0], volumeData.emissive[ch * 3 + 1], volumeData.emissive[ch * 3 + 2]);
}

DEV inline Vec3f NormalizedGradient4ch(const Vec3f& P, const cudaVolume& volumeData, int ch)
{
	Vec3f Gradient;

	Gradient.x = (GetNormalizedIntensity4ch(P + (gGradientDeltaX), volumeData, ch) - GetNormalizedIntensity4ch(P - (gGradientDeltaX), volumeData, ch)) * gInvGradientDelta;
	Gradient.y = (GetNormalizedIntensity4ch(P + (gGradientDeltaY), volumeData, ch) - GetNormalizedIntensity4ch(P - (gGradientDeltaY), volumeData, ch)) * gInvGradientDelta;
	Gradient.z = (GetNormalizedIntensity4ch(P + (gGradientDeltaZ), volumeData, ch) - GetNormalizedIntensity4ch(P - (gGradientDeltaZ), volumeData, ch)) * gInvGradientDelta;

	return Normalize(Gradient);
}

DEV float GradientMagnitude(const Vec3f& P, cudaTextureObject_t texGradientMagnitude)
{
	return ((float)SHRT_MAX * tex3D<float>(texGradientMagnitude, P.x * gInvAaBbMax.x, P.y * gInvAaBbMax.y, P.z * gInvAaBbMax.z));
}

DEV int NearestLight(const CudaLighting& lighting, CRay R, CColorXyz& LightColor, Vec3f& Pl, float* pPdf = NULL)
{
	int Hit = -1;
	
	float T = 0.0f;

	CRay RayCopy = R;

	float Pdf = 0.0f;

	//printf(" LIGHTS %d", lighting.m_NoLights);
	for (int i = 0; i < lighting.m_NoLights; i++)
	{
		if (lighting.m_Lights[i].Intersect(RayCopy, T, LightColor, NULL, &Pdf))
		{
			Pl		= R(T);
			Hit		= i;
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
	const float SmallestMaxT = fminf(fminf(MaxT.x, MaxT.y), fminf(MaxT.x, MaxT.z));

	*pNearT = LargestMinT;
	*pFarT	= SmallestMaxT;

	return SmallestMaxT > LargestMinT;
}

DEV CColorXyza CumulativeMovingAverage(const CColorXyza& A, const CColorXyza& Ax, const int& N)
{
//	if (gNoIterations == 0)
//		return CColorXyza(0.0f);

	 return A + ((Ax - A) / max((float)N, 1.0f));
}