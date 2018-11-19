#pragma once

#include <cuda_runtime.h>

// this gives the ability to read a float4 and then loop over its elements.
typedef union {
	float4 vec;
	float a[4];
} f4;

DEV float GetNormalizedIntensityMax4ch(const float3& P, const cudaVolume& volumeData, int& ch)
{
	//float factor = (tex3D<float>(volumeData.volumeTexture[5], P.x * gInvAaBbMax.x, P.y * gInvAaBbMax.y, P.z * gInvAaBbMax.z));
	//factor = (factor > 0) ? 1.0 : 0.0;
	f4 intensity;
	intensity.vec = ((float)UINT16_MAX * tex3D<float4>(volumeData.volumeTexture[0], P.x * gInvAaBbMax.x, P.y * gInvAaBbMax.y, P.z * gInvAaBbMax.z));
	float maxIn = 0;
	for (int i = 0; i < min(volumeData.nChannels, 4); ++i) {
		// 0..1
		intensity.a[i] = (intensity.a[i] - volumeData.intensityMin[i]) / (volumeData.intensityMax[i] - volumeData.intensityMin[i]);
		// transform through LUT
		intensity.a[i] = tex1D<float>(volumeData.lutTexture[i], intensity.a[i]);
		if (intensity.a[i] > maxIn) {
			maxIn = intensity.a[i];
			ch = i;
		}
	}
	return maxIn; // *factor;
}

DEV float GetNormalizedIntensity4ch(const float3& P, const cudaVolume& volumeData, int ch)
{
	f4 intensity;
	intensity.vec = ((float)UINT16_MAX * tex3D<float4>(volumeData.volumeTexture[0], P.x * gInvAaBbMax.x, P.y * gInvAaBbMax.y, P.z * gInvAaBbMax.z));
	// select channel
	float intensityf = intensity.a[ch];
	intensityf = (intensityf - volumeData.intensityMin[ch]) / (volumeData.intensityMax[ch] - volumeData.intensityMin[ch]);
	//intensity = tex1D<float>(volumeData.lutTexture[ch], intensity);
	return intensityf;
}

DEV f4 GetIntensity4ch(const float3& P, const cudaVolume& volumeData)
{
	//float factor = (tex3D<float>(volumeData.volumeTexture[5], P.x * gInvAaBbMax.x, P.y * gInvAaBbMax.y, P.z * gInvAaBbMax.z));
	//factor = (factor > 0) ? 1.0 : 0.0;
	f4 intensity;
	intensity.vec = make_float4(0, 0, 0, 0);
	intensity.vec = ((float)UINT16_MAX * tex3D<float4>(volumeData.volumeTexture[0], P.x * gInvAaBbMax.x, P.y * gInvAaBbMax.y, P.z * gInvAaBbMax.z));
	for (int i = 0; i < min(volumeData.nChannels, 4); ++i) {
		// 0..1
		intensity.a[i] = (intensity.a[i] - volumeData.intensityMin[i]) / (volumeData.intensityMax[i] - volumeData.intensityMin[i]);
		// transform through LUT
		intensity.a[i] = tex1D<float>(volumeData.lutTexture[i], intensity.a[i]);
	}
	return intensity;
}


DEV float GetOpacity(const float& NormalizedIntensity, const cudaVolume& volumeData, int ch)
{
	// apply lut
	float Intensity = NormalizedIntensity * volumeData.opacity[ch];
	//float Intensity = tex1D<float>(texLut, NormalizedIntensity);
	return Intensity;
}

DEV float GetBlendedOpacity(const cudaVolume& volumeData, f4 intensity)
{
	float sum = 0.0;
	int n = min(volumeData.nChannels, 4);
	for (int i = 0; i < n; ++i) {
		sum += intensity.a[i];
	}
	return sum / n;
}

DEV CColorRgbHdr GetDiffuseN(const float& NormalizedIntensity, const cudaVolume& volumeData, int ch)
{
	return CColorRgbHdr(volumeData.diffuse[ch * 3 + 0], volumeData.diffuse[ch * 3 + 1], volumeData.diffuse[ch * 3 + 2]);
}

DEV CColorRgbHdr GetBlendedDiffuse(const cudaVolume& volumeData, f4 intensity)
{
	CColorRgbHdr retval(0.0, 0.0, 0.0);
	float sum = 0.0;
	for (int i = 0; i < min(volumeData.nChannels, 4); ++i) {
		// blend.
		retval.r += volumeData.diffuse[i * 3 + 0] * intensity.a[i];
		retval.g += volumeData.diffuse[i * 3 + 1] * intensity.a[i];
		retval.b += volumeData.diffuse[i * 3 + 2] * intensity.a[i];
		sum += intensity.a[i];
	}
	retval.r /= sum;
	retval.g /= sum;
	retval.b /= sum;
	return retval;
}

DEV CColorRgbHdr GetSpecularN(const float& NormalizedIntensity, const cudaVolume& volumeData, int ch)
{
	return CColorRgbHdr(volumeData.specular[ch * 3 + 0], volumeData.specular[ch * 3 + 1], volumeData.specular[ch * 3 + 2]);
}

DEV CColorRgbHdr GetBlendedSpecular(const cudaVolume& volumeData, f4 intensity)
{
	CColorRgbHdr retval(0.0, 0.0, 0.0);
	float sum = 0.0;
	for (int i = 0; i < min(volumeData.nChannels, 4); ++i) {
		// blend.
		retval.r += volumeData.specular[i * 3 + 0] * intensity.a[i];
		retval.g += volumeData.specular[i * 3 + 1] * intensity.a[i];
		retval.b += volumeData.specular[i * 3 + 2] * intensity.a[i];
		sum += intensity.a[i];
	}
	retval.r /= sum;
	retval.g /= sum;
	retval.b /= sum;
	return retval;
}

DEV CColorRgbHdr GetEmissionN(const float& NormalizedIntensity, const cudaVolume& volumeData, int ch)
{
	return CColorRgbHdr(volumeData.emissive[ch * 3 + 0], volumeData.emissive[ch * 3 + 1], volumeData.emissive[ch * 3 + 2]);
}

DEV CColorRgbHdr GetBlendedEmission(const cudaVolume& volumeData, f4 intensity)
{
	CColorRgbHdr retval(0.0, 0.0, 0.0);
	float sum = 0.0;
	for (int i = 0; i < min(volumeData.nChannels, 4); ++i) {
		// blend.
		retval.r += volumeData.emissive[i * 3 + 0] * intensity.a[i];
		retval.g += volumeData.emissive[i * 3 + 1] * intensity.a[i];
		retval.b += volumeData.emissive[i * 3 + 2] * intensity.a[i];
		sum += intensity.a[i];
	}
	retval.r /= sum;
	retval.g /= sum;
	retval.b /= sum;
	return retval;
}

DEV float GetRoughnessN(const float& NormalizedIntensity, const cudaVolume& volumeData, int ch)
{
	return volumeData.roughness[ch];
}

DEV float GetBlendedRoughness(f4 intensity, const cudaVolume& volumeData)
{
	float retval = 0.0;
	float sum = 0.0;
	for (int i = 0; i < min(volumeData.nChannels, 4); ++i) {
		// blend.
		retval += volumeData.roughness[i] * intensity.a[i];
		sum += intensity.a[i];
	}
	return retval / sum;
}

DEV inline float3 NormalizedGradient4ch(const float3& P, const cudaVolume& volumeData, int ch)
{
	float3 Gradient;

	Gradient.x = (GetNormalizedIntensity4ch(P + (gGradientDeltaX), volumeData, ch) - GetNormalizedIntensity4ch(P - (gGradientDeltaX), volumeData, ch)) * gInvGradientDelta;
	Gradient.y = (GetNormalizedIntensity4ch(P + (gGradientDeltaY), volumeData, ch) - GetNormalizedIntensity4ch(P - (gGradientDeltaY), volumeData, ch)) * gInvGradientDelta;
	Gradient.z = (GetNormalizedIntensity4ch(P + (gGradientDeltaZ), volumeData, ch) - GetNormalizedIntensity4ch(P - (gGradientDeltaZ), volumeData, ch)) * gInvGradientDelta;

	return normalize(Gradient);
}
// note that gInvGradientDelta is maxpixeldim of volume
// gGradientDeltaX,Y,Z is 1/X,Y,Z of volume
DEV inline float3 Gradient4ch(const float3& P, const cudaVolume& volumeData, int ch)
{
	float3 Gradient;

	Gradient.x = (GetNormalizedIntensity4ch(P + (gGradientDeltaX), volumeData, ch) - GetNormalizedIntensity4ch(P - (gGradientDeltaX), volumeData, ch)) * gInvGradientDelta;
	Gradient.y = (GetNormalizedIntensity4ch(P + (gGradientDeltaY), volumeData, ch) - GetNormalizedIntensity4ch(P - (gGradientDeltaY), volumeData, ch)) * gInvGradientDelta;
	Gradient.z = (GetNormalizedIntensity4ch(P + (gGradientDeltaZ), volumeData, ch) - GetNormalizedIntensity4ch(P - (gGradientDeltaZ), volumeData, ch)) * gInvGradientDelta;

	return Gradient;
}

DEV float GradientMagnitude(const float3& P, cudaTextureObject_t texGradientMagnitude)
{
	return ((float)UINT16_MAX * tex3D<float>(texGradientMagnitude, P.x * gInvAaBbMax.x, P.y * gInvAaBbMax.y, P.z * gInvAaBbMax.z));
}

DEV int NearestLight(const CudaLighting& lighting, CRay R, CColorXyz& LightColor, float3& Pl, float* pPdf = NULL)
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
	const float3 InvR		= make_float3(1.0f, 1.0f, 1.0f) / R.m_D;
	const float3 BottomT		= InvR * (make_float3(gClippedAaBbMin.x, gClippedAaBbMin.y, gClippedAaBbMin.z) - R.m_O);
	const float3 TopT		= InvR * (make_float3(gClippedAaBbMax.x, gClippedAaBbMax.y, gClippedAaBbMax.z) - R.m_O);
	const float3 MinT		= fminf(TopT, BottomT);
	const float3 MaxT		= fmaxf(TopT, BottomT);
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
