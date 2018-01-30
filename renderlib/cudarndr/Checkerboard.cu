#include "Checkerboard.cuh"

#include "CudaUtilities.h"
//#include "Geometry.h"

CD int			gFilmWidth1;
CD int			gFilmHeight1;

#define KRNL_CH_BLOCK_W		16
#define KRNL_CH_BLOCK_H		8
#define KRNL_CH_BLOCK_SIZE	KRNL_CH_BLOCK_W * KRNL_CH_BLOCK_H

KERNEL void KrnlCh(cudaSurfaceObject_t surfaceObj)
{
	const int X 	= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;
	//const int TID	= threadIdx.y * blockDim.x + threadIdx.x;

	// bounds check.
	if (X >= gFilmWidth1 || Y >= gFilmHeight1)
		return;

	uchar4 sample = make_uchar4(0,255,0,255);
	surf2Dwrite(sample, surfaceObj, X*4, Y);
}

void Checkerboard(cudaSurfaceObject_t surfaceObj, int w, int h)
{
	// init some input vars for kernel
	HandleCudaError(cudaMemcpyToSymbol(gFilmWidth1, &w, sizeof(int)));
	HandleCudaError(cudaMemcpyToSymbol(gFilmHeight1, &h, sizeof(int)));

	// launch kernel
	const dim3 KernelBlock(KRNL_CH_BLOCK_W, KRNL_CH_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)w / (float)KernelBlock.x), (int)ceilf((float)h / (float)KernelBlock.y));
	KrnlCh<<<KernelGrid, KernelBlock>>>(surfaceObj);
	
	// wait for end kernel
	cudaDeviceSynchronize();
	HandleCudaKernelError(cudaGetLastError(), "Checkerboard");
}