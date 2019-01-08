#pragma once

#include "Defines.h"
#include "Enumerations.h"

#include <algorithm>
#include <math.h>

using namespace std;

HOD inline float
Lerp(float t, float v1, float v2)
{
  return (1.f - t) * v1 + t * v2;
}

HOD inline float
clamp2(float v, float a, float b)
{
  return max(a, min(v, b));
}

HOD inline void
swap(int& a, int& b)
{
  int t = a;
  a = b;
  b = t;
}

HOD inline void
swap(float& a, float& b)
{
  float t = a;
  a = b;
  b = t;
}

HOD inline void
Swap(float* pF1, float* pF2)
{
  const float TempFloat = *pF1;

  *pF1 = *pF2;
  *pF2 = TempFloat;
}

HOD inline void
Swap(float& F1, float& F2)
{
  const float TempFloat = F1;

  F1 = F2;
  F2 = TempFloat;
}

HOD inline void
Swap(int* pI1, int* pI2)
{
  const int TempInt = *pI1;

  *pI1 = *pI2;
  *pI2 = TempInt;
}

HOD inline void
Swap(int& I1, int& I2)
{
  const int TempInt = I1;

  I1 = I2;
  I2 = TempInt;
}
class CColorXyz;

HOD inline void
XYZToRGB(const float xyz[3], float rgb[3])
{
  rgb[0] = 3.240479f * xyz[0] - 1.537150f * xyz[1] - 0.498535f * xyz[2];
  rgb[1] = -0.969256f * xyz[0] + 1.875991f * xyz[1] + 0.041556f * xyz[2];
  rgb[2] = 0.055648f * xyz[0] - 0.204043f * xyz[1] + 1.057311f * xyz[2];
}

HOD inline void
RGBToXYZ(const float rgb[3], float xyz[3])
{
  xyz[0] = 0.412453f * rgb[0] + 0.357580f * rgb[1] + 0.180423f * rgb[2];
  xyz[1] = 0.212671f * rgb[0] + 0.715160f * rgb[1] + 0.072169f * rgb[2];
  xyz[2] = 0.019334f * rgb[0] + 0.119193f * rgb[1] + 0.950227f * rgb[2];
}

static const int gNoSamplesSpectrumXYZ = 3;

CD static float YWeight[gNoSamplesSpectrumXYZ] = { 0.212671f, 0.715160f, 0.072169f };

class CColorXyz
{
public:
  enum EType
  {
    Reflectance,
    Illuminant
  };

  // SampledSpectrum Public Methods
  HOD CColorXyz(float v = 0.f)
  {
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      c[i] = v;
  }

  HOD CColorXyz(float x, float y, float z)
  {
    c[0] = x;
    c[1] = y;
    c[2] = z;
  }

  HOD CColorXyz& operator+=(const CColorXyz& s2)
  {
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      c[i] += s2.c[i];
    return *this;
  }

  HOD CColorXyz operator+(const CColorXyz& s2) const
  {
    CColorXyz ret = *this;
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      ret.c[i] += s2.c[i];
    return ret;
  }

  HOD CColorXyz operator-(const CColorXyz& s2) const
  {
    CColorXyz ret = *this;
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      ret.c[i] -= s2.c[i];
    return ret;
  }

  HOD CColorXyz operator/(const CColorXyz& s2) const
  {
    CColorXyz ret = *this;
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      ret.c[i] /= s2.c[i];
    return ret;
  }

  HOD CColorXyz operator*(const CColorXyz& sp) const
  {
    CColorXyz ret = *this;
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      ret.c[i] *= sp.c[i];
    return ret;
  }

  HOD CColorXyz& operator*=(const CColorXyz& sp)
  {
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      c[i] *= sp.c[i];
    return *this;
  }

  HOD CColorXyz operator*(float a) const
  {
    CColorXyz ret = *this;
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      ret.c[i] *= a;
    return ret;
  }

  HOD CColorXyz& operator*=(float a)
  {
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      c[i] *= a;
    return *this;
  }

  HOD friend inline CColorXyz operator*(float a, const CColorXyz& s) { return s * a; }

  HOD CColorXyz operator/(float a) const
  {
    CColorXyz ret = *this;
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      ret.c[i] /= a;
    return ret;
  }

  HOD CColorXyz& operator/=(float a)
  {
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      c[i] /= a;
    return *this;
  }

  HOD bool operator==(const CColorXyz& sp) const
  {
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      if (c[i] != sp.c[i])
        return false;
    return true;
  }

  HOD bool operator!=(const CColorXyz& sp) const { return !(*this == sp); }

  HOD float operator[](int i) const { return c[i]; }

  HOD float operator[](int i) { return c[i]; }

  // ToDo: Add description
  HOD CColorXyz& operator=(const CColorXyz& Other)
  {
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      c[i] = Other.c[i];

    // By convention, always return *this
    return *this;
  }

  HOD bool IsBlack() const
  {
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      if (c[i] != 0.)
        return false;
    return true;
  }

  HOD CColorXyz Clamp(float low = 0, float high = INF_MAX) const
  {
    CColorXyz ret;
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      ret.c[i] = clamp2(c[i], low, high);
    return ret;
  }

  DEV float y() const
  {
    float v = 0.;
    for (int i = 0; i < gNoSamplesSpectrumXYZ; i++)
      v += YWeight[i] * c[i];
    return v;
  }

  HOD void ToRGB(float rgb[3] /* , float* pCIEX, float* pCIEY, float* pCIEZ */) const
  {
    rgb[0] = c[0];
    rgb[1] = c[1];
    rgb[2] = c[2];

    XYZToRGB(c, rgb);
  }

  // 	RGBSpectrum ToRGBSpectrum() const;

  HOD static CColorXyz FromXYZ(float r, float g, float b)
  {
    CColorXyz L;

    L.c[0] = r;
    L.c[1] = g;
    L.c[2] = b;

    return L;
  }

  HOD static CColorXyz FromRGB(float r, float g, float b)
  {
    const float CoeffX[3] = { 0.4124f, 0.3576f, 0.1805f };
    const float CoeffY[3] = { 0.2126f, 0.7152f, 0.0722f };
    const float CoeffZ[3] = { 0.0193f, 0.1192f, 0.9505f };

    float XYZ[3];

    XYZ[0] = CoeffX[0] * r + CoeffX[1] * g + CoeffX[2] * b;

    XYZ[1] = CoeffY[0] * r + CoeffY[1] * g + CoeffY[2] * b;

    XYZ[2] = CoeffZ[0] * r + CoeffZ[1] * g + CoeffZ[2] * b;

    return CColorXyz::FromXYZ(XYZ[0], XYZ[1], XYZ[2]);
  }

  // 	static CSampledSpectrum FromXYZ(const float xyz[3], SpectrumType type = SPECTRUM_REFLECTANCE)
  // 	{
  // 		float rgb[3];
  // 		XYZToRGB(xyz, rgb);
  // 		return FromRGB(rgb, type);
  // 	}

  // 	CSampledSpectrum(const RGBSpectrum &r, SpectrumType type = SPECTRUM_REFLECTANCE);

public:
  float c[3];
};

class CSpectrumSample
{
public:
  float m_C;
  int m_Index;

  // ToDo: Add description
  HOD CSpectrumSample(void)
  {
    m_C = 0.0f;
    m_Index = 0;
  };

  // ToDo: Add description
  HOD ~CSpectrumSample(void) {}

  // ToDo: Add description
  HOD CSpectrumSample& operator=(const CSpectrumSample& Other)
  {
    m_C = Other.m_C;
    m_Index = Other.m_Index;

    // By convention, always return *this
    return *this;
  }
};

/*
Spectrum FromXYZ(float x, float y, float z) {
        float c[3];
        c[0] =  3.240479f * x + -1.537150f * y + -0.498535f * z;
        c[1] = -0.969256f * x +  1.875991f * y +  0.041556f * z;
        c[2] =  0.055648f * x + -0.204043f * y +  1.057311f * z;
        return Spectrum(c);
}*/

// static inline HOD SpectrumXYZ MakeSpectrum(void)
// { SpectrumXYZ s; s.c[0] = 0.0f; s.c[1] = 0.0f; s.c[2] = 0.0f; return s;				} static inline
// HOD SpectrumXYZ MakeSpectrum(const float& r, const float& g, const float& b)
// { SpectrumXYZ s; s.c[0] = r; s.c[1] = g; s.c[2] = b; return s; } static inline HOD SpectrumXYZ MakeSpectrum(const
// float& rgb)																{ SpectrumXYZ s; s.c[0] =
// rgb; s.c[1] = rgb; s.c[2] = rgb; return s;					}

class CColorXyza
{
public:
  enum EType
  {
    Reflectance,
    Illuminant
  };

  // SampledSpectrum Public Methods
  HOD CColorXyza(float v = 0.f)
  {
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      c[i] = v;
  }

  HOD CColorXyza(float x, float y, float z)
  {
    c[0] = x;
    c[1] = y;
    c[2] = z;
  }

  HOD CColorXyza& operator+=(const CColorXyza& s2)
  {
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      c[i] += s2.c[i];
    return *this;
  }

  HOD CColorXyza operator+(const CColorXyza& s2) const
  {
    CColorXyza ret = *this;
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      ret.c[i] += s2.c[i];
    return ret;
  }

  HOD CColorXyza operator-(const CColorXyza& s2) const
  {
    CColorXyza ret = *this;
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      ret.c[i] -= s2.c[i];
    return ret;
  }

  HOD CColorXyza operator/(const CColorXyza& s2) const
  {
    CColorXyza ret = *this;
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      ret.c[i] /= s2.c[i];
    return ret;
  }

  HOD CColorXyza operator*(const CColorXyza& sp) const
  {
    CColorXyza ret = *this;
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      ret.c[i] *= sp.c[i];
    return ret;
  }

  HOD CColorXyza& operator*=(const CColorXyza& sp)
  {
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      c[i] *= sp.c[i];
    return *this;
  }

  HOD CColorXyza operator*(float a) const
  {
    CColorXyza ret = *this;
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      ret.c[i] *= a;
    return ret;
  }

  HOD CColorXyza& operator*=(float a)
  {
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      c[i] *= a;
    return *this;
  }

  HOD friend inline CColorXyza operator*(float a, const CColorXyza& s) { return s * a; }

  HOD CColorXyza operator/(float a) const
  {
    CColorXyza ret = *this;
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      ret.c[i] /= a;
    return ret;
  }

  HOD CColorXyza& operator/=(float a)
  {
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      c[i] /= a;
    return *this;
  }

  HOD bool operator==(const CColorXyza& sp) const
  {
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      if (c[i] != sp.c[i])
        return false;
    return true;
  }

  HOD bool operator!=(const CColorXyza& sp) const { return !(*this == sp); }

  HOD float operator[](int i) const { return c[i]; }

  HOD float operator[](int i) { return c[i]; }

  // ToDo: Add description
  HOD CColorXyza& operator=(const CColorXyza& Other)
  {
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      c[i] = Other.c[i];

    // By convention, always return *this
    return *this;
  }

  HOD bool IsBlack() const
  {
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      if (c[i] != 0.)
        return false;
    return true;
  }

  HOD CColorXyza Clamp(float low = 0, float high = INF_MAX) const
  {
    CColorXyza ret;
    for (int i = 0; i < gNoSamplesSpectrumXYZ; ++i)
      ret.c[i] = clamp2(c[i], low, high);
    return ret;
  }

  DEV float y() const
  {
    float v = 0.;
    for (int i = 0; i < gNoSamplesSpectrumXYZ; i++)
      v += YWeight[i] * c[i];
    return v;
  }

  HOD void ToRGB(float rgb[3], float* pCIEX, float* pCIEY, float* pCIEZ) const
  {
    rgb[0] = c[0];
    rgb[1] = c[1];
    rgb[2] = c[2];

    XYZToRGB(c, rgb);
  }

  // 	RGBSpectrum ToRGBSpectrum() const;

  HOD static CColorXyza FromXYZ(float r, float g, float b)
  {
    CColorXyza L;

    L.c[0] = r;
    L.c[1] = g;
    L.c[2] = b;

    return L;
  }

  HOD static CColorXyza FromRGB(float r, float g, float b)
  {
    const float CoeffX[3] = { 0.4124f, 0.3576f, 0.1805f };
    const float CoeffY[3] = { 0.2126f, 0.7152f, 0.0722f };
    const float CoeffZ[3] = { 0.0193f, 0.1192f, 0.9505f };

    float XYZ[3];

    XYZ[0] = CoeffX[0] * r + CoeffX[1] * g + CoeffX[2] * b;

    XYZ[1] = CoeffY[0] * r + CoeffY[1] * g + CoeffY[2] * b;

    XYZ[2] = CoeffZ[0] * r + CoeffZ[1] * g + CoeffZ[2] * b;

    return CColorXyza::FromXYZ(XYZ[0], XYZ[1], XYZ[2]);
  }

  // 	static CSampledSpectrum FromXYZ(const float xyz[3], SpectrumType type = SPECTRUM_REFLECTANCE)
  // 	{
  // 		float rgb[3];
  // 		XYZToRGB(xyz, rgb);
  // 		return FromRGB(rgb, type);
  // 	}

  // 	CSampledSpectrum(const RGBSpectrum &r, SpectrumType type = SPECTRUM_REFLECTANCE);

public:
  float c[4];
};

// Colors
#define CLR_RAD_BLACK CColorXyz(0.0f)
#define CLR_RAD_WHITE CColorXyz(1.0f)
#define CLR_RAD_RED CColorXyz(1.0f, 0.0f, 0.0f)
#define CLR_RAD_GREEN CColorXyz(0.0f, 1.0f, 0.0f)
#define CLR_RAD_BLUE CColorXyz(0.0f, 0.0f, 1.0f)
#define SPEC_BLACK CColorXyz(0.0f)
#define SPEC_GRAY_10 CColorXyz(1.0f)
#define SPEC_GRAY_20 CColorXyz(1.0f)
#define SPEC_GRAY_30 CColorXyz(1.0f)
#define SPEC_GRAY_40 CColorXyz(1.0f)
#define SPEC_GRAY_50 CColorXyz(0.5f)
#define SPEC_GRAY_60 CColorXyz(1.0f)
#define SPEC_GRAY_70 CColorXyz(1.0f)
#define SPEC_GRAY_80 CColorXyz(1.0f)
#define SPEC_GRAY_90 CColorXyz(1.0f)
#define SPEC_WHITE CColorXyz(1.0f)
#define SPEC_CYAN CColorXyz(1.0f)
#define SPEC_RED CColorXyz(1.0f, 0.0f, 0.0f)

#define XYZA_BLACK CColorXyza(0.0f)
