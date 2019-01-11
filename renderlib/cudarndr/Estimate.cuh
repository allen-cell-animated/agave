#pragma once

#include "Geometry.h"

#define KRNL_ESTIMATE_BLOCK_W 32
#define KRNL_ESTIMATE_BLOCK_H 32
#define KRNL_ESTIMATE_BLOCK_SIZE (KRNL_ESTIMATE_BLOCK_W * KRNL_ESTIMATE_BLOCK_H)

KERNEL void
KrnlEstimate(float* pView, float* pViewAccum)
{
  const int X = blockIdx.x * blockDim.x + threadIdx.x;
  const int Y = blockIdx.y * blockDim.y + threadIdx.y;

  if (X >= gFilmWidth || Y >= gFilmHeight)
    return;

  // pixel offset of this thread
  int pixoffset = Y * (gFilmWidth) + (X);
  int floatoffset = pixoffset * 4;

  float4 ac;
  ac.x = pViewAccum[floatoffset];
  ac.y = pViewAccum[floatoffset + 1];
  ac.z = pViewAccum[floatoffset + 2];
  ac.w = pViewAccum[floatoffset + 3];
  float4 sample;
  sample.x = pView[floatoffset];
  sample.y = pView[floatoffset + 1];
  sample.z = pView[floatoffset + 2];
  sample.w = pView[floatoffset + 3];
  float4 cx = CumulativeMovingAverage(ac, sample, gNoIterations);
  pViewAccum[floatoffset] = cx.x;
  pViewAccum[floatoffset + 1] = cx.y;
  pViewAccum[floatoffset + 2] = cx.z;
  // what is the correct alpha?
  pViewAccum[floatoffset + 3] = cx.w;
}

void
Estimate(int res_x, int res_y, float* pView, float* pViewAccum)
{
  const dim3 KernelBlock(KRNL_ESTIMATE_BLOCK_W, KRNL_ESTIMATE_BLOCK_H);
  const dim3 KernelGrid((int)ceilf((float)res_x / (float)KernelBlock.x),
                        (int)ceilf((float)res_y / (float)KernelBlock.y));
  // const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.GetWidth() / (float)KernelBlock.x),
  // (int)ceilf((float)pScene->m_Camera.m_Film.GetHeight() / (float)KernelBlock.y));

  KrnlEstimate<<<KernelGrid, KernelBlock>>>(pView, pViewAccum);
  HandleCudaKernelError(cudaGetLastError(), "Compute Estimate");
  cudaDeviceSynchronize();
  HandleCudaKernelError(cudaGetLastError(), "Compute Estimate");
}
