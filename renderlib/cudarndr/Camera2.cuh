#pragma once

// enough data to generate a camera ray
// float3 is used to have a struct that has no ctor, so that this can be stored in __constant__ memory.
struct CudaCamera
{
  float m_FocalDistance;
  float3 m_From;
  float3 m_N;
  float3 m_U;
  float3 m_V;
  float m_ApertureSize;
  float m_InvScreen[2];
  float m_Screen[2][2];
};
