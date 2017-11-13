#pragma once

#include "Geometry.h"
#include "Variance.h"

#define KRNL_TM_BLOCK_W		8
#define KRNL_TM_BLOCK_H		8
#define KRNL_TM_BLOCK_SIZE	KRNL_TM_BLOCK_W * KRNL_TM_BLOCK_H

KERNEL void KrnlToneMap(float* inbuf, cudaSurfaceObject_t surfaceObj)
{
	const int X = blockIdx.x * blockDim.x + threadIdx.x;
	const int Y = blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= gFilmWidth || Y >= gFilmHeight)
		return;

	int pixoffset = Y*(gFilmWidth)+(X);

	float4 sample = reinterpret_cast<float4*>(inbuf)[pixoffset];

	sample.x = __saturatef(1.0f - expf(-(sample.x * gInvExposure)));
	sample.y = __saturatef(1.0f - expf(-(sample.y * gInvExposure)));
	sample.z = __saturatef(1.0f - expf(-(sample.z * gInvExposure)));
	sample.w = __saturatef(sample.w);

	uchar4 pixel = make_uchar4(sample.x*255.0, sample.y*255.0, sample.z*255.0, sample.w*255.0);
	surf2Dwrite(pixel, surfaceObj, X * 4, Y);
}

void ToneMap(float* inbuf, cudaSurfaceObject_t surfaceObj, int w, int h)
{
	const dim3 KernelBlock(KRNL_TM_BLOCK_W, KRNL_TM_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)w / (float)KernelBlock.x), (int)ceilf((float)h / (float)KernelBlock.y));
	KrnlToneMap<<<KernelGrid, KernelBlock>>>(inbuf, surfaceObj);

	cudaDeviceSynchronize();
	HandleCudaKernelError(cudaGetLastError(), "Tone Map");
}

