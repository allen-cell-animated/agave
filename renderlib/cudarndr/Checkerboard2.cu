#include "Checkerboard2.cuh"

#include "CudaUtilities.h"

CD int gFilmWidth2;
CD int gFilmHeight2;

#define KRNL_CH_BLOCK_W 16
#define KRNL_CH_BLOCK_H 8
#define KRNL_CH_BLOCK_SIZE (KRNL_CH_BLOCK_W * KRNL_CH_BLOCK_H)

KERNEL void
KrnlCh(float* outbuf)
{
  const int X = blockIdx.x * blockDim.x + threadIdx.x;
  const int Y = blockIdx.y * blockDim.y + threadIdx.y;
  // const int TID	= threadIdx.y * blockDim.x + threadIdx.x;

  // bounds check.
  if (X >= gFilmWidth2 || Y >= gFilmHeight2)
    return;

  int pixoffset = Y * (gFilmWidth2) + (X);
  int floatoffset = pixoffset * 4;
  outbuf[floatoffset] = 0.75;
  outbuf[floatoffset + 1] = 0.0;
  outbuf[floatoffset + 2] = 0.25;
  outbuf[floatoffset + 3] = 1.0;
}

void
Checkerboard2(float* outbuf, int w, int h)
{
  // init some input vars for kernel
  HandleCudaError(cudaMemcpyToSymbol(gFilmWidth2, &w, sizeof(int)));
  HandleCudaError(cudaMemcpyToSymbol(gFilmHeight2, &h, sizeof(int)));

  // launch kernel
  const dim3 KernelBlock(KRNL_CH_BLOCK_W, KRNL_CH_BLOCK_H);
  const dim3 KernelGrid((int)ceilf((float)w / (float)KernelBlock.x), (int)ceilf((float)h / (float)KernelBlock.y));
  KrnlCh<<<KernelGrid, KernelBlock>>>(outbuf);

  // wait for end kernel
  cudaDeviceSynchronize();
  HandleCudaKernelError(cudaGetLastError(), "Checkerboard2");
}
