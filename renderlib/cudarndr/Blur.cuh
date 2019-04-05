#pragma once

#include "Geometry.h"
#include "CudaUtilities.h"

#define KRNL_BLUR_BLOCK_W		16
#define KRNL_BLUR_BLOCK_H		8
#define KRNL_BLUR_BLOCK_SIZE	KRNL_BLUR_BLOCK_W * KRNL_BLUR_BLOCK_H

KERNEL void KrnlBlurH(CCudaView* pView)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;
	const int TID	= threadIdx.y * blockDim.x + threadIdx.x;

	if (X >= gFilmWidth || Y >= gFilmHeight)
		return;

	const int X0 = max((int)ceilf(X - gFilterWidth), 0);
	const int X1 = min((int)floorf(X + gFilterWidth), (int)gFilmWidth - 1);

	CColorXyza Sum;

	__shared__ float FW[KRNL_BLUR_BLOCK_SIZE];
	__shared__ float SumW[KRNL_BLUR_BLOCK_SIZE];

	__syncthreads();

	FW[TID]		= 0.0f;
	SumW[TID]	= 0.0f;

	for (int x = X0; x <= X1; x++)
	{
		FW[TID] = gFilterWeights[(int)fabs((float)x - X)];

		Sum			+= pView->m_FrameEstimateXyza.Get(x, Y) * FW[TID];
		SumW[TID]	+= FW[TID];
	}

	if (SumW[TID] > 0.0f) {
		CColorXyza cx(Sum / SumW[TID]);
		pView->m_FrameBlurXyza.Set(cx, X, Y);
	}
	else {
		CColorXyza cx(0.0f);
		pView->m_FrameBlurXyza.Set(cx, X, Y);
	}
}

KERNEL void KrnlBlurV(CCudaView* pView)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;
	const int TID	= threadIdx.y * blockDim.x + threadIdx.x;

	if (X >= gFilmWidth || Y >= gFilmHeight)
		return;

	const int Y0 = max((int)ceilf (Y - gFilterWidth), 0);
	const int Y1 = min((int)floorf(Y + gFilterWidth), gFilmHeight - 1);

	CColorXyza Sum;

	__shared__ float FW[KRNL_BLUR_BLOCK_SIZE];
	__shared__ float SumW[KRNL_BLUR_BLOCK_SIZE];

	__syncthreads();

	FW[TID]		= 0.0f;
	SumW[TID]	= 0.0f;

	for (int y = Y0; y <= Y1; y++)
	{
		FW[TID] = gFilterWeights[(int)fabs((float)y - Y)];

		Sum			+= pView->m_FrameBlurXyza.Get(X, y) * FW[TID];
		SumW[TID]	+= FW[TID];
	}

	if (SumW[TID] > 0.0f) {
		CColorXyza cx(Sum / SumW[TID]);
		pView->m_FrameEstimateXyza.Set(cx, X, Y);
	}
	else {
		CColorXyza cx(0.0f);
		pView->m_FrameEstimateXyza.Set(cx, X, Y);
	}
}

void Blur(int res_x, int res_y, CCudaView* pDevView)
{
	const dim3 KernelBlock(KRNL_BLUR_BLOCK_W, KRNL_BLUR_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)res_x / (float)KernelBlock.x), (int)ceilf((float)res_y / (float)KernelBlock.y));

	KrnlBlurH<<<KernelGrid, KernelBlock>>>(pDevView);
	cudaDeviceSynchronize();
	HandleCudaKernelError(cudaGetLastError(), "Blur Estimate H");
	
	KrnlBlurV<<<KernelGrid, KernelBlock>>>(pDevView);
	cudaDeviceSynchronize();
	HandleCudaKernelError(cudaGetLastError(), "Blur Estimate V");
}
