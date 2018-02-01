#pragma once

#include "Geometry.h"
#include "CudaUtilities.h"

#define KRNL_SS_BLOCK_W		8
#define KRNL_SS_BLOCK_H		8
#define KRNL_SS_BLOCK_SIZE	KRNL_SS_BLOCK_W * KRNL_SS_BLOCK_H

DEV inline bool SampleDistanceRM(CRay& R, CRNG& RNG, Vec3f& Ps, const cudaVolume& volumedata)
{
	const int TID = threadIdx.y * blockDim.x + threadIdx.x;

	__shared__ float MinT[KRNL_SS_BLOCK_SIZE];
	__shared__ float MaxT[KRNL_SS_BLOCK_SIZE];

	if (!IntersectBox(R, &MinT[TID], &MaxT[TID]))
		return false;

	MinT[TID] = max(MinT[TID], R.m_MinT);
	MaxT[TID] = min(MaxT[TID], R.m_MaxT);

	// ray march along the ray's projected path and keep an average sigmaT value.
	// When the distance has become greater than the average sigmaT value given by -log(RandomFloat[0, 1]) / averageSigmaT 
	// then that would be considered the interaction position.

	const float S	= -log(RNG.Get1()) / gDensityScale;
	float Sum		= 0.0f;
	float SigmaT	= 0.0f;

	MinT[TID] += RNG.Get1() * gStepSize;
	int ch = 0;
	while (Sum < S)
	{
		Ps = R.m_O + MinT[TID] * R.m_D;

		if (MinT[TID] > MaxT[TID])
			return false;
		
		SigmaT = gDensityScale * GetOpacity(GetNormalizedIntensityMax4ch(Ps, volumedata, ch));

		Sum			+= SigmaT * gStepSize;
		MinT[TID]	+= gStepSize;
	}

	return true;
}

DEV inline bool FreePathRM(CRay& R, CRNG& RNG, const cudaVolume& volumedata)
{
	const int TID = threadIdx.y * blockDim.x + threadIdx.x;

	__shared__ float MinT[KRNL_SS_BLOCK_SIZE];
	__shared__ float MaxT[KRNL_SS_BLOCK_SIZE];
	__shared__ Vec3f Ps[KRNL_SS_BLOCK_SIZE];

	if (!IntersectBox(R, &MinT[TID], &MaxT[TID]))
		return false;

	MinT[TID] = max(MinT[TID], R.m_MinT);
	MaxT[TID] = min(MaxT[TID], R.m_MaxT);

	const float S	= -log(RNG.Get1()) / gDensityScale;
	float Sum		= 0.0f;
	float SigmaT	= 0.0f;

	MinT[TID] += RNG.Get1() * gStepSizeShadow;
	int ch = 0;
	while (Sum < S)
	{
		Ps[TID] = R.m_O + MinT[TID] * R.m_D;

		if (MinT[TID] > MaxT[TID])
			return false;
		
		SigmaT = gDensityScale * GetOpacity(GetNormalizedIntensityMax4ch(Ps[TID], volumedata, ch));

		Sum			+= SigmaT * gStepSizeShadow;
		MinT[TID]	+= gStepSizeShadow;
	}

	return true;
}

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
	while (T < MaxT)
	{
		Ps = R.m_O + T * R.m_D;

		if (GetOpacity(GetNormalizedIntensityMax4ch(Ps, volumedata, ch)) > 0.0f)
			return true;

		T += gStepSize;
	}

	return false;
}