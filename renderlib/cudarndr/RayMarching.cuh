#pragma once

#include "Geometry.h"
#include "CudaUtilities.h"

#define KRNL_SS_BLOCK_W		8
#define KRNL_SS_BLOCK_H		8
#define KRNL_SS_BLOCK_SIZE	KRNL_SS_BLOCK_W * KRNL_SS_BLOCK_H

DEV inline bool SampleDistanceRM(CRay& R, CRNG& RNG, Vec3f& Ps, const cudaVolume& volumedata)
{
	float MinT;
	float MaxT;

	if (!IntersectBox(R, &MinT, &MaxT))
		return false;

	MinT = max(MinT, R.m_MinT);
	MaxT = min(MaxT, R.m_MaxT);

	// ray march along the ray's projected path and keep an average sigmaT value.
	// When the distance has become greater than the average sigmaT value given by -log(RandomFloat[0, 1]) / averageSigmaT 
	// then that would be considered the interaction position.

	// sigmaT = sigmaA + sigmaS = absorption coeff + scattering coeff = extinction coeff

	// Beer-Lambert law: transmittance T(t) = exp(-sigmaT*t)
	// importance sampling the exponential function to produce a free path distance S
	// the PDF is p(t) = sigmaT * exp(-sigmaT * t)
	// S is the free-path distance = -ln(1-zeta)/sigmaT where zeta is a random variable
	const float S	= -log(RNG.Get1()) / gDensityScale;  // note that ln(x:0..1) is negative

	// density scale 0... S --> 0..inf.  Low density means randomly sized ray paths
	// density scale inf... S --> 0.   High density means short ray paths!
	float Sum		= 0.0f;
	float SigmaT	= 0.0f; // accumulated extinction along ray march

	MinT += RNG.Get1() * gStepSize;
	int ch = 0;
	float intensity = 0.0;
	// ray march until we have traveled S (or hit the maxT of the ray)
	while (Sum < S)
	{
		Ps = R.m_O + MinT * R.m_D;

		if (MinT > MaxT)
			return false;
		
		intensity = GetNormalizedIntensityMax4ch(Ps, volumedata, ch);
		SigmaT = gDensityScale * GetOpacity(intensity, volumedata, ch);
		//SigmaT = gDensityScale * GetBlendedOpacity(volumedata, GetIntensity4ch(Ps, volumedata));

		Sum			+= SigmaT * gStepSize;
		MinT	+= gStepSize;
	}

	// Ps is the point
	return true;
}

// "shadow ray" using gStepSizeShadow, test whether it can exit the volume or not
DEV inline bool FreePathRM(CRay& R, CRNG& RNG, const cudaVolume& volumedata)
{
	float MinT;
	float MaxT;
	Vec3f Ps;

	if (!IntersectBox(R, &MinT, &MaxT))
		return false;

	MinT = max(MinT, R.m_MinT);
	MaxT = min(MaxT, R.m_MaxT);

	const float S	= -log(RNG.Get1()) / gDensityScale;
	float Sum		= 0.0f;
	float SigmaT	= 0.0f;

	MinT += RNG.Get1() * gStepSizeShadow;
	int ch = 0;
	float intensity = 0.0;
	while (Sum < S)
	{
		Ps = R.m_O + MinT * R.m_D;

		if (MinT > MaxT)
			return false;
		
		intensity = GetNormalizedIntensityMax4ch(Ps, volumedata, ch);
		SigmaT = gDensityScale * GetOpacity(intensity, volumedata, ch);
		// SigmaT = gDensityScale * GetBlendedOpacity(volumedata, GetIntensity4ch(Ps, volumedata));

		Sum			+= SigmaT * gStepSizeShadow;
		MinT	+= gStepSizeShadow;
	}

	return true;
}

// does the ray encounter any nonzero intensity?
DEV inline bool NearestIntersection(CRay R, const cudaVolume& volumedata, float& T)
{
	float MinT = 0.0f, MaxT = 0.0f;

	if (!IntersectBox(R, &MinT, &MaxT))
		return false;

	MinT = max(MinT, R.m_MinT);
	MaxT = min(MaxT, R.m_MaxT);

	Vec3f Ps; 

	T = MinT;
	int ch = 0;
	float intensity = 0.0;
	while (T < MaxT)
	{
		Ps = R.m_O + T * R.m_D;

		intensity = GetNormalizedIntensityMax4ch(Ps, volumedata, ch);
		if (GetOpacity(intensity, volumedata, ch) > 0.0f) {
		//if (GetBlendedOpacity(volumedata, GetIntensity4ch(Ps, volumedata)) > 0.0f) {
			return true;
		}

		T += gStepSize;
	}

	return false;
}
