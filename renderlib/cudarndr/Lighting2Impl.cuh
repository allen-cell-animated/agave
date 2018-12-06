#include "Lighting2.cuh"

#include "helper_math.cuh"

#include "MonteCarlo.cuh"
#include "Sample.cuh"

DEV CColorXyz
ToXYZ(const float3& f)
{
  return CColorXyz::FromRGB(f.x, f.y, f.z);
}
DEV CColorRgbHdr
toRGB(const float3& f)
{
  return CColorRgbHdr(f.x, f.y, f.z);
}

// Samples the light
DEV CColorXyz
CudaLight::SampleL(const float3& P, CRay& Rl, float& Pdf, CLightingSample& LS) const
{
  CColorXyz L = SPEC_BLACK;

  if (m_T == 0) {
    Rl.m_O = (m_P + ((-0.5f + LS.m_LightSample.m_Pos.x) * m_Width * m_U) +
              ((-0.5f + LS.m_LightSample.m_Pos.y) * m_Height * m_V));
    Rl.m_D = normalize(P - Rl.m_O);
    L = dot(Rl.m_D, m_N) > 0.0f ? Le(make_float2(0.0f)) : SPEC_BLACK;
    Pdf = AbsDot(Rl.m_D, m_N) > 0.0f ? DistanceSquared(P, Rl.m_O) / (AbsDot(Rl.m_D, m_N) * m_Area) : 0.0f;
  }

  if (m_T == 1) {
    Rl.m_O = (m_P) + m_SkyRadius * UniformSampleSphere(LS.m_LightSample.m_Pos);
    Rl.m_D = normalize(P - Rl.m_O);
    L = Le(make_float2(1.0f) - 2.0f * LS.m_LightSample.m_Pos);
    Pdf = powf(m_SkyRadius, 2.0f) / m_Area;
  }

  Rl.m_MinT = 0.0f;
  Rl.m_MaxT = Length(P - Rl.m_O);

  return L;
}

// Intersect ray with light
DEV bool
CudaLight::Intersect(CRay& R, float& T, CColorXyz& L, float2* pUV, float* pPdf) const
{
  if (m_T == 0) {
    // Compute projection
    const float DotN = dot(R.m_D, m_N);

    // Rays is co-planar with light surface
    if (DotN >= 0.0f)
      return false;

    // Compute hit distance
    T = (-m_Distance - dot(R.m_O, m_N)) / DotN;

    // Intersection is in ray's negative direction
    if (T < R.m_MinT || T > R.m_MaxT)
      return false;

    // Determine position on light
    const float3 Pl = R(T);

    // Vector from point on area light to center of area light
    const float3 Wl = Pl - (m_P);

    // Compute texture coordinates
    const float2 UV = make_float2(dot(Wl, m_U), dot(Wl, m_V));

    // Check if within bounds of light surface
    if (UV.x > m_HalfWidth || UV.x < -m_HalfWidth || UV.y > m_HalfHeight || UV.y < -m_HalfHeight)
      return false;

    R.m_MaxT = T;

    if (pUV)
      *pUV = UV;

    if (DotN < 0.0f)
      L = ToXYZ(m_Color) / m_Area;
    else
      L = SPEC_BLACK;

    if (pPdf)
      *pPdf = DistanceSquared(R.m_O, Pl) / (DotN * m_Area);

    return true;
  }

  if (m_T == 1) {
    T = m_SkyRadius;

    // Intersection is in ray's negative direction
    if (T < R.m_MinT || T > R.m_MaxT)
      return false;

    R.m_MaxT = T;

    float2 UV = make_float2(SphericalPhi(R.m_D) * INV_TWO_PI_F, SphericalTheta(R.m_D) * INV_PI_F);

    L = Le(make_float2(1.0f) - 2.0f * UV);

    if (pPdf)
      *pPdf = powf(m_SkyRadius, 2.0f) / m_Area;

    return true;
  }

  return false;
}

DEV float
CudaLight::Pdf(const float3& P, const float3& Wi) const
{
  CColorXyz L;
  // float2 UV;
  float Pdf = 1.0f;

  CRay Rl = CRay(P, Wi, 0.0f, INF_MAX);

  if (m_T == 0) {
    float T = 0.0f;

    if (!Intersect(Rl, T, L, NULL, &Pdf))
      return 0.0f;

    return powf(T, 2.0f) / (AbsDot(m_N, -1.0 * Wi) * m_Area);
  }

  if (m_T == 1) {
    return powf(m_SkyRadius, 2.0f) / m_Area;
  }

  return 0.0f;
}

// c1 and c2 as rgb colors
DEV inline CColorRgbHdr
Lerp(float T, const float3& C1, const float3& C2)
{
  const float OneMinusT = 1.0f - T;
  return CColorRgbHdr(OneMinusT * C1.x + T * C2.x, OneMinusT * C1.y + T * C2.y, OneMinusT * C1.z + T * C2.z);
}

DEV CColorXyz
CudaLight::Le(const float2& UV) const
{
  if (m_T == 0)
    return CColorXyz::FromRGB(m_Color.x, m_Color.y, m_Color.z) / m_Area;

  if (m_T == 1) {
    if (UV.y > 0.0f)
      return Lerp(fabs(UV.y), m_ColorMiddle, m_ColorTop).ToXYZ();
    else
      return Lerp(fabs(UV.y), m_ColorMiddle, m_ColorBottom).ToXYZ();
  }

  return SPEC_BLACK;
}
