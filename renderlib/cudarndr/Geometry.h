#pragma once

#include "Spectrum.h"

//#include "helper_math.cuh"

#include <algorithm>
#include <math.h>
#include <stdio.h>

class CColorRgbHdr;
class Vec4i;
class CColorXyz;

class CColorRgbHdr
{
public:
  HOD CColorRgbHdr(void)
  {
    r = 0.0f;
    g = 0.0f;
    b = 0.0f;
  }

  HOD CColorRgbHdr(const float& r, const float& g, const float& b)
  {
    this->r = r;
    this->g = g;
    this->b = b;
  }

  HOD CColorRgbHdr(const float& rgb)
  {
    r = rgb;
    g = rgb;
    b = rgb;
  }

  HOD CColorRgbHdr& operator=(const CColorRgbHdr& p)
  {
    r = p.r;
    g = p.g;
    b = p.b;

    return *this;
  }

  HOD CColorRgbHdr& operator=(const CColorXyz& S);

  HOD CColorRgbHdr& operator+=(CColorRgbHdr& p)
  {
    r += p.r;
    g += p.g;
    b += p.b;

    return *this;
  }

  HOD CColorRgbHdr operator*(float f) const { return CColorRgbHdr(r * f, g * f, b * f); }

  HOD CColorRgbHdr& operator*=(float f)
  {
    for (int i = 0; i < 3; i++)
      (&r)[i] *= f;

    return *this;
  }

  HOD CColorRgbHdr operator/(float f) const
  {
    float inv = 1.0f / f;
    return CColorRgbHdr(r * inv, g * inv, b * inv);
  }

  HOD CColorRgbHdr& operator/=(float f)
  {
    float inv = 1.f / f;
    r *= inv;
    g *= inv;
    b *= inv;
    return *this;
  }

  HOD float operator[](int i) const { return (&r)[i]; }

  HOD float operator[](int i) { return (&r)[i]; }

  HOD bool Black(void) { return r == 0.0f && g == 0.0f && b == 0.0f; }

  HOD CColorRgbHdr Pow(float e) { return CColorRgbHdr(powf(r, e), powf(g, e), powf(b, e)); }

  HOD void FromXYZ(float x, float y, float z)
  {
    const float rWeight[3] = { 3.240479f, -1.537150f, -0.498535f };
    const float gWeight[3] = { -0.969256f, 1.875991f, 0.041556f };
    const float bWeight[3] = { 0.055648f, -0.204043f, 1.057311f };

    r = rWeight[0] * x + rWeight[1] * y + rWeight[2] * z;

    g = gWeight[0] * x + gWeight[1] * y + gWeight[2] * z;

    b = bWeight[0] * x + bWeight[1] * y + bWeight[2] * z;
  }

  HOD CColorXyz ToXYZ(void) const { return CColorXyz::FromRGB(r, g, b); }

  HOD CColorXyza ToXYZA(void) const { return CColorXyza::FromRGB(r, g, b); }

  void PrintSelf(void) { printf("[%g, %g, %g]\n", r, g, b); }

  float r;
  float g;
  float b;
};

class Vec4i
{
public:
  HOD Vec4i(void)
  {
    this->x = 0;
    this->y = 0;
    this->z = 0;
    this->w = 0;
  }

  HOD Vec4i(const int& x, const int& y, const int& z, const int& w)
  {
    this->x = x;
    this->y = y;
    this->z = z;
    this->w = w;
  }

  HOD Vec4i(const int& xyzw)
  {
    this->x = xyzw;
    this->y = xyzw;
    this->z = xyzw;
    this->w = xyzw;
  }

  int x, y, z, w;
};

static HOD CColorRgbHdr operator*(const CColorRgbHdr& v, const float& f)
{
  return CColorRgbHdr(f * v.r, f * v.g, f * v.b);
};
static HOD CColorRgbHdr operator*(const float& f, const CColorRgbHdr& v)
{
  return CColorRgbHdr(f * v.r, f * v.g, f * v.b);
};
static HOD CColorRgbHdr operator*(const CColorRgbHdr& p1, const CColorRgbHdr& p2)
{
  return CColorRgbHdr(p1.r * p2.r, p1.g * p2.g, p1.b * p2.b);
};
static HOD CColorRgbHdr
operator+(const CColorRgbHdr& a, const CColorRgbHdr& b)
{
  return CColorRgbHdr(a.r + b.r, a.g + b.g, a.b + b.b);
};

HOD inline CColorRgbHdr&
CColorRgbHdr::operator=(const CColorXyz& S)
{
  r = S.c[0];
  g = S.c[1];
  b = S.c[2];

  return *this;
}

// reflect
inline HOD float
Fmaxf(float a, float b)
{
  return a > b ? a : b;
}

inline HOD float
Fminf(float a, float b)
{
  return a < b ? a : b;
}

inline HOD float
Clamp(float f, float a, float b)
{
  return Fmaxf(a, Fminf(f, b));
}

HOD inline CColorRgbHdr
Lerp(float T, const CColorRgbHdr& C1, const CColorRgbHdr& C2)
{
  const float OneMinusT = 1.0f - T;
  return CColorRgbHdr(OneMinusT * C1.r + T * C2.r, OneMinusT * C1.g + T * C2.g, OneMinusT * C1.b + T * C2.b);
}

HOD inline CColorXyz
Lerp(float T, const CColorXyz& C1, const CColorXyz& C2)
{
  const float OneMinusT = 1.0f - T;
  return CColorXyz(OneMinusT * C1.c[0] + T * C2[0], OneMinusT * C1.c[0] + T * C2[0], OneMinusT * C1.c[0] + T * C2[0]);
}
