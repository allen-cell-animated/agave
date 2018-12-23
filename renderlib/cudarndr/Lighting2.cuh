#pragma once

#include "Geometry.h"

class CLightingSample;

struct CudaLight
{
  float m_Theta;
  float m_Phi;
  float m_Width;
  float m_InvWidth;
  float m_HalfWidth;
  float m_InvHalfWidth;
  float m_Height;
  float m_InvHeight;
  float m_HalfHeight;
  float m_InvHalfHeight;
  float m_Distance;
  float m_SkyRadius;
  float3 m_P;
  float3 m_Target;
  float3 m_N;
  float3 m_U;
  float3 m_V;
  float m_Area;
  float m_AreaPdf;
  float3 m_Color;
  float3 m_ColorTop;
  float3 m_ColorMiddle;
  float3 m_ColorBottom;
  int m_T;

  // Samples the light
  DEV CColorXyz SampleL(const float3& P, CRay& Rl, float& Pdf, CLightingSample& LS) const;
  // Intersect ray with light
  DEV bool Intersect(CRay& R, float& T, CColorXyz& L, float2* pUV = NULL, float* pPdf = NULL) const;
  DEV float Pdf(const float3& P, const float3& Wi) const;
  DEV CColorXyz Le(const float2& UV) const;
};
#define MAX_CUDA_LIGHTS 4
struct CudaLighting
{
  int m_NoLights;
  CudaLight m_Lights[MAX_CUDA_LIGHTS];
};
