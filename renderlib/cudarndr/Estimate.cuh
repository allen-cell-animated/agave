#pragma once

#include "Geometry.h"

#define KRNL_ESTIMATE_BLOCK_W		32
#define KRNL_ESTIMATE_BLOCK_H		32
#define KRNL_ESTIMATE_BLOCK_SIZE	KRNL_ESTIMATE_BLOCK_W * KRNL_ESTIMATE_BLOCK_H

KERNEL void KrnlEstimate(float* pView, float* pViewAccum)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= gFilmWidth || Y >= gFilmHeight)
		return;

	// pixel offset of this thread
	int pixoffset = Y*(gFilmWidth)+(X);
	int floatoffset = pixoffset * 4;

	CColorXyza ac;
	ac.c[0] = pViewAccum[floatoffset];
	ac.c[1] = pViewAccum[floatoffset + 1];
	ac.c[2] = pViewAccum[floatoffset + 2];
	ac.c[3] = pViewAccum[floatoffset + 3];
	CColorXyza sample;
	sample.c[0] = pView[floatoffset];
	sample.c[1] = pView[floatoffset + 1];
	sample.c[2] = pView[floatoffset + 2];
	sample.c[3] = pView[floatoffset + 3];
	CColorXyza cx = CumulativeMovingAverage(ac, sample, gNoIterations);
	pViewAccum[floatoffset] = cx.c[0];
	pViewAccum[floatoffset + 1] = cx.c[1];
	pViewAccum[floatoffset + 2] = cx.c[2];
	// what is the correct alpha?
	pViewAccum[floatoffset + 3] = 1.0f;// cx.c[3];
}

void Estimate(CScene* pScene, CScene* pDevScene, float* pView, float* pViewAccum)
{
	const dim3 KernelBlock(KRNL_ESTIMATE_BLOCK_W, KRNL_ESTIMATE_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResX() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResY() / (float)KernelBlock.y));
	//const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.GetWidth() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.GetHeight() / (float)KernelBlock.y));

	KrnlEstimate<<<KernelGrid, KernelBlock>>>(pView, pViewAccum);
	HandleCudaKernelError(cudaGetLastError(), "Compute Estimate");
	cudaDeviceSynchronize();
	HandleCudaKernelError(cudaGetLastError(), "Compute Estimate");
}