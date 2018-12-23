#pragma once

#include "Geometry.h"

#include "RNG.cuh"

class CLightSample
{
public:
  float2 m_Pos;
  float m_Component;

  HOD CLightSample(void)
  {
    m_Pos = make_float2(0.0f);
    m_Component = 0.0f;
  }

  HOD CLightSample& operator=(const CLightSample& Other)
  {
    m_Pos = Other.m_Pos;
    m_Component = Other.m_Component;

    return *this;
  }

  DEV void LargeStep(CRNG& Rnd)
  {
    m_Pos = Rnd.Get2();
    m_Component = Rnd.Get1();
  }
};

class CBrdfSample
{
public:
  float m_Component;
  float2 m_Dir;

  HOD CBrdfSample(void)
  {
    m_Component = 0.0f;
    m_Dir = make_float2(0.0f);
  }

  HOD CBrdfSample(const float& Component, const float2& Dir)
  {
    m_Component = Component;
    m_Dir = Dir;
  }

  HOD CBrdfSample& operator=(const CBrdfSample& Other)
  {
    m_Component = Other.m_Component;
    m_Dir = Other.m_Dir;

    return *this;
  }

  DEV void LargeStep(CRNG& Rnd)
  {
    m_Component = Rnd.Get1();
    m_Dir = Rnd.Get2();
  }
};

class CLightingSample
{
public:
  CBrdfSample m_BsdfSample;
  CLightSample m_LightSample;
  float m_LightNum;

  HOD CLightingSample(void) { m_LightNum = 0.0f; }

  HOD CLightingSample& operator=(const CLightingSample& Other)
  {
    m_BsdfSample = Other.m_BsdfSample;
    m_LightNum = Other.m_LightNum;
    m_LightSample = Other.m_LightSample;

    return *this;
  }

  DEV void LargeStep(CRNG& Rnd)
  {
    m_BsdfSample.LargeStep(Rnd);
    m_LightSample.LargeStep(Rnd);

    m_LightNum = Rnd.Get1();
  }
};

class CCameraSample
{
public:
  float2 m_ImageXY;
  float2 m_LensUV;

  DEV CCameraSample(void)
  {
    m_ImageXY = make_float2(0.0f);
    m_LensUV = make_float2(0.0f);
  }

  DEV CCameraSample& operator=(const CCameraSample& Other)
  {
    m_ImageXY = Other.m_ImageXY;
    m_LensUV = Other.m_LensUV;

    return *this;
  }

  DEV void LargeStep(float2& ImageUV, float2& LensUV, const int& X, const int& Y, const int& KernelSize)
  {
    m_ImageXY = make_float2(X + ImageUV.x, Y + ImageUV.y);
    m_LensUV = LensUV;
  }
};
