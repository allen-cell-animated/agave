#include "GLPTVolumeShader.h"

#include "AppScene.h"
#include "BoundingBox.h"
#include "CCamera.h"
#include "DenoiseParams.h"
#include "ImageXYZC.h"
#include "ImageXyzcGpu.h"
#include "Logging.h"

#include <gl/Util.h>
#include <glm.h>

#include <iostream>
#include <sstream>

GLPTVolumeShader::GLPTVolumeShader()
  : GLShaderProgram()
  , m_vshader()
  , m_fshader()
{
  m_vshader = new GLShader(GL_VERTEX_SHADER);
  m_vshader->compileSourceCode(R"(
#version 400 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 uv;
out vec2 vUv;

void main()
{
  vUv = uv;
  gl_Position = vec4( position, 1.0 );
}

	)");

  if (!m_vshader->isCompiled()) {
    LOG_ERROR << "GLPTVolumeShader: Failed to compile vertex shader\n" << m_vshader->log();
  }

  m_fshader = new GLShader(GL_FRAGMENT_SHADER);
  const char* fsPiece1 = R"(
#version 400 core

#define PI (3.1415926535897932384626433832795)
#define PI_OVER_2 (1.57079632679489661923)
#define PI_OVER_4 (0.785398163397448309616)
#define INV_PI (1.0/PI)
#define INV_2_PI (0.5/PI)
#define INV_4_PI (0.25/PI)

const vec3 BLACK = vec3(0,0,0);
const vec3 WHITE = vec3(1.0,1.0,1.0);
const int ShaderType_Brdf = 0;
const int ShaderType_Phase = 1;


in vec2 vUv;
out vec4 out_FragColor;

struct Camera {
  vec3 m_from;
  vec3 m_U, m_V, m_N;
  vec4 m_screen;  // left, right, bottom, top
  vec2 m_invScreen;  // 1/w, 1/h
  float m_focalDistance;
  float m_apertureSize;
  float m_isPerspective;
};

uniform Camera gCamera;

struct Light {
  float   m_theta;
  float   m_phi;
  float   m_width;
  float   m_halfWidth;
  float   m_height;
  float   m_halfHeight;
  float   m_distance;
  float   m_skyRadius;
  vec3    m_P;
  vec3    m_target;
  vec3    m_N;
  vec3    m_U;
  vec3    m_V;
  float   m_area;
  float   m_areaPdf;
  vec3    m_color;
  vec3    m_colorTop;
  vec3    m_colorMiddle;
  vec3    m_colorBottom;
  int     m_T;
};
const int NUM_LIGHTS = 2;
uniform Light gLights[2];

uniform vec3 gClippedAaBbMin;
uniform vec3 gClippedAaBbMax;
uniform float gDensityScale;
uniform float gStepSize;
uniform float gStepSizeShadow;
uniform sampler3D volumeTexture;
uniform vec3 gPosToUVW;
uniform int g_nChannels;
uniform int gShadingType;
uniform vec3 gGradientDeltaX;
uniform vec3 gGradientDeltaY;
uniform vec3 gGradientDeltaZ;
uniform float gInvGradientDelta;
uniform float gGradientFactor;
uniform float uShowLights;

// per channel
uniform sampler2D g_lutTexture[4];
uniform sampler2D g_colormapTexture[4];
uniform vec4 g_intensityMax;
uniform vec4 g_intensityMin;
uniform vec4 g_lutMax;
uniform vec4 g_lutMin;
uniform vec4 g_labels;
uniform float g_opacity[4];
uniform vec3 g_emissive[4];
uniform vec3 g_diffuse[4];
uniform vec3 g_specular[4];
uniform float g_roughness[4];

// compositing / progressive render
uniform float uFrameCounter;
uniform float uSampleCounter;
uniform vec2 uResolution;
uniform sampler2D tPreviousTexture;

// from https://www.shadertoy.com/view/4ssXRX
// uvec3 pcg3d(uvec3 v) {

//     v = v * 1664525u + 1013904223u;

//     v.x += v.y*v.z;
//     v.y += v.z*v.x;
//     v.z += v.x*v.y;

//     v ^= v >> 16u;

//     v.x += v.y*v.z;
//     v.y += v.z*v.x;
//     v.z += v.x*v.y;

//     return v;
// }
// vec3 pcg3d_f( vec3 v )
// {
//     return (1.0/float(0xffffffffu)) * vec3(pcg3d( uvec3(floatBitsToUint(v.x),
//                   			 							floatBitsToUint(v.y),
//                   			 							floatBitsToUint(v.z)) ));
// }
// float nrand( vec3 n )
// {
//     return pcg3d_f(n).x;
// }
// float n1rand( in vec2 uv, in float t )
// {
// 	float nrnd0 = nrand( vec3(uv,t) + 0.07 );
// 	return nrnd0;
// }

// from iq https://www.shadertoy.com/view/4tXyWN
float rand( inout uvec2 seed )
{
  seed += uvec2(1);
      uvec2 q = 1103515245U * ( (seed >> 1U) ^ (seed.yx) );
      uint  n = 1103515245U * ( (q.x) ^ (q.y >> 3U) );
  return float(n) * (1.0 / float(0xffffffffU));
}

vec3 XYZtoRGB(vec3 xyz) {
  return vec3(
    3.240479f*xyz[0] - 1.537150f*xyz[1] - 0.498535f*xyz[2],
    -0.969256f*xyz[0] + 1.875991f*xyz[1] + 0.041556f*xyz[2],
    0.055648f*xyz[0] - 0.204043f*xyz[1] + 1.057311f*xyz[2]
  );
}

vec3 RGBtoXYZ(vec3 rgb) {
  return vec3(
    0.412453f*rgb[0] + 0.357580f*rgb[1] + 0.180423f*rgb[2],
    0.212671f*rgb[0] + 0.715160f*rgb[1] + 0.072169f*rgb[2],
    0.019334f*rgb[0] + 0.119193f*rgb[1] + 0.950227f*rgb[2]
  );
}
)";

  const char* fsPiece2 = R"(
vec3 getUniformSphereSample(in vec2 U)
{
  float z = 1.f - 2.f * U.x;
  float r = sqrt(max(0.f, 1.f - z*z));
  float phi = 2.f * PI * U.y;
  float x = r * cos(phi);
  float y = r * sin(phi);
  return vec3(x, y, z);
}

float SphericalPhi(in vec3 Wl)
{
  float p = atan(Wl.z, Wl.x);
  return (p < 0.f) ? p + 2.f * PI : p;
}

float SphericalTheta(in vec3 Wl)
{
  return acos(clamp(Wl.y, -1.f, 1.f));
}

bool SameHemisphere(in vec3 Ww1, in vec3 Ww2)
{
   return (Ww1.z * Ww2.z) > 0.0f;
}

vec2 getConcentricDiskSample(in vec2 U)
{
  float r, theta;
  // Map uniform random numbers to [-1,1]^2
  float sx = 2.0 * U.x - 1.0;
  float sy = 2.0 * U.y - 1.0;
  // Map square to (r,theta)
  // Handle degeneracy at the origin

  if (sx == 0.0 && sy == 0.0)
  {
    return vec2(0.0f, 0.0f);
  }

  if (sx >= -sy)
  {
    if (sx > sy)
    {
      // Handle first region of disk
      r = sx;
      if (sy > 0.0)
        theta = sy/r;
      else
        theta = 8.0f + sy/r;
    }
    else
    {
      // Handle second region of disk
      r = sy;
      theta = 2.0f - sx/r;
    }
  }
  else
  {
    if (sx <= sy)
    {
      // Handle third region of disk
      r = -sx;
      theta = 4.0f - sy/r;
    }
    else
    {
      // Handle fourth region of disk
      r = -sy;
      theta = 6.0f + sx/r;
    }
  }

  theta *= PI_OVER_4;

  return vec2(r*cos(theta), r*sin(theta));
}

vec3 getCosineWeightedHemisphereSample(in vec2 U)
{
  vec2 ret = getConcentricDiskSample(U);
  return vec3(ret.x, ret.y, sqrt(max(0.f, 1.f - ret.x * ret.x - ret.y * ret.y)));
}

struct Ray {
  vec3 m_O;
  vec3 m_D;
  float m_MinT, m_MaxT;
};

Ray newRay(in vec3 o, in vec3 d) {
  return Ray(o, d, 0.0, 1500000.0);
}

Ray newRayT(in vec3 o, in vec3 d, in float t0, in float t1) {
  return Ray(o, d, t0, t1);
}

vec3 rayAt(Ray r, float t) {
  return r.m_O + t*r.m_D;
}

Ray GenerateCameraRay(in Camera cam, in vec2 Pixel, in vec2 ApertureRnd)
{
  vec2 ScreenPoint;

  // m_screen: x:left, y:right, z:bottom, w:top
  ScreenPoint.x = cam.m_screen.x + (cam.m_invScreen.x * Pixel.x);
  ScreenPoint.y = cam.m_screen.z + (cam.m_invScreen.y * Pixel.y);

  vec3 RayO = cam.m_from;
  if (cam.m_isPerspective == 0.0) {
    RayO += (ScreenPoint.x * cam.m_U) + (ScreenPoint.y * cam.m_V);
  }

  vec3 RayD = normalize(cam.m_N + (ScreenPoint.x * cam.m_U) + (ScreenPoint.y * cam.m_V));
  if (cam.m_isPerspective == 0.0) {
    RayD = cam.m_N;
  }

  if (cam.m_apertureSize != 0.0f)
  {
    vec2 LensUV = cam.m_apertureSize * getConcentricDiskSample(ApertureRnd);

    vec3 LI = cam.m_U * LensUV.x + cam.m_V * LensUV.y;
    RayO += LI;
    RayD = normalize((RayD * cam.m_focalDistance) - LI);
  }

  return newRay(RayO, RayD);
}

bool IntersectBox(in Ray R, out float pNearT, out float pFarT)
{
  vec3 invR		= vec3(1.0f, 1.0f, 1.0f) / R.m_D;
  vec3 bottomT		= invR * (vec3(gClippedAaBbMin.x, gClippedAaBbMin.y, gClippedAaBbMin.z) - R.m_O);
  vec3 topT		= invR * (vec3(gClippedAaBbMax.x, gClippedAaBbMax.y, gClippedAaBbMax.z) - R.m_O);
  vec3 minT		= min(topT, bottomT);
  vec3 maxT		= max(topT, bottomT);
  float largestMinT = max(max(minT.x, minT.y), max(minT.x, minT.z));
  float smallestMaxT = min(min(maxT.x, maxT.y), min(maxT.x, maxT.z));

  pNearT = largestMinT;
  pFarT	= smallestMaxT;

  return pFarT > pNearT;
}

vec3 PtoVolumeTex(vec3 p) {
  // center of volume is 0.5*extents
  // this needs to return a number in 0..1 range, so just rescale to bounds.
  return p * gPosToUVW;
}

const float UINT16_MAX = 65535.0;
float GetNormalizedIntensityMax4ch(in vec3 P, out int ch)
{
  vec4 intensity = UINT16_MAX * texture(volumeTexture, PtoVolumeTex(P));

  float maxIn = 0.0;
  ch = 0;

  // relative to min/max for each channel
  intensity = (intensity - g_intensityMin) / (g_intensityMax - g_intensityMin);
  intensity.x = texture(g_lutTexture[0], vec2(intensity.x, 0.5)).x * pow(g_opacity[0], 4.0);
  intensity.y = texture(g_lutTexture[1], vec2(intensity.y, 0.5)).x * pow(g_opacity[1], 4.0);
  intensity.z = texture(g_lutTexture[2], vec2(intensity.z, 0.5)).x * pow(g_opacity[2], 4.0);
  intensity.w = texture(g_lutTexture[3], vec2(intensity.w, 0.5)).x * pow(g_opacity[3], 4.0);

  // take the high value of the 4 channels
  for (int i = 0; i < min(g_nChannels, 4); ++i) {
    if (intensity[i] > maxIn) {
      maxIn = intensity[i];
      ch = i;
    }
  }
  return maxIn; // *factor;
}
float GetNormalizedIntensityRnd4ch(in vec3 P, out int ch, inout uvec2 seed)
{
  vec4 intensity = UINT16_MAX * texture(volumeTexture, PtoVolumeTex(P));

  float maxIn = 0.0;
  ch = 0;

  // relative to min/max for each channel
  intensity = (intensity - g_intensityMin) / (g_intensityMax - g_intensityMin);

  // take a random value of the 4 channels
  // TODO weight this based on the post-LUT 4-channel intensities?
  float r = rand(seed)*min(float(g_nChannels), 4.0);
  ch = int(r);

  float retval = texture(g_lutTexture[ch], vec2(intensity[ch], 0.5)).x * pow(g_opacity[ch], 4.0);

  return retval;
}
float GetNormalizedIntensityRnd4ch_weighted(in vec3 P, out int ch, inout uvec2 seed)
{
  vec4 intensity = UINT16_MAX * texture(volumeTexture, PtoVolumeTex(P));

  ch = 0;

  // relative to min/max for each channel
  intensity = (intensity - g_intensityMin) / (g_intensityMax - g_intensityMin);
  intensity.x = texture(g_lutTexture[0], vec2(intensity.x, 0.5)).x * pow(g_opacity[0], 4.0);
  intensity.y = texture(g_lutTexture[1], vec2(intensity.y, 0.5)).x * pow(g_opacity[1], 4.0);
  intensity.z = texture(g_lutTexture[2], vec2(intensity.z, 0.5)).x * pow(g_opacity[2], 4.0);
  intensity.w = texture(g_lutTexture[3], vec2(intensity.w, 0.5)).x * pow(g_opacity[3], 4.0);

  // ensure 0 for nonexistent channels?
  float sum = intensity.x + intensity.y + intensity.z + intensity.w;
  // take a random value of the 4 channels
  float r = rand(seed)*sum;
  float cum = 0;
  float retval = 0;
  for (int i = 0; i < min(g_nChannels, 4); ++i) {
    cum = cum + intensity[i];
    if (r < cum) {
      ch = i;
      retval = intensity[i];
      break;
    }
  }
  return retval;
}

float GetNormalizedIntensity(in vec3 P, in int ch)
{
  float intensity = UINT16_MAX * texture(volumeTexture, PtoVolumeTex(P))[ch];
  intensity = (intensity - g_intensityMin[ch]) / (g_intensityMax[ch] - g_intensityMin[ch]);
  intensity = texture(g_lutTexture[ch], vec2(intensity, 0.5)).x;
  return intensity;
}

float GetNormalizedIntensity4ch(vec3 P, int ch)
{
  vec4 intensity = UINT16_MAX * texture(volumeTexture, PtoVolumeTex(P));
  // select channel
  float intensityf = intensity[ch];
  intensityf = (intensityf - g_intensityMin[ch]) / (g_intensityMax[ch] - g_intensityMin[ch]);
  //intensityf = texture(g_lutTexture[ch], vec2(intensityf, 0.5)).x;

  return intensityf;
}

float GetRawIntensity(vec3 P, int ch)
{
  return texture(volumeTexture, PtoVolumeTex(P))[ch];
}

// note that gInvGradientDelta is maxpixeldim of volume
// gGradientDeltaX,Y,Z is 1/X,Y,Z of volume
vec3 Gradient4ch(vec3 P, int ch)
{
  vec3 Gradient;

  //Gradient.x = (GetRawIntensity(P + (gGradientDeltaX), ch) - GetRawIntensity(P - (gGradientDeltaX), ch)) * gInvGradientDelta;
  //Gradient.y = (GetRawIntensity(P + (gGradientDeltaY), ch) - GetRawIntensity(P - (gGradientDeltaY), ch)) * gInvGradientDelta;
  //Gradient.z = (GetRawIntensity(P + (gGradientDeltaZ), ch) - GetRawIntensity(P - (gGradientDeltaZ), ch)) * gInvGradientDelta;

  Gradient.x = (GetNormalizedIntensity(P + (gGradientDeltaX), ch) - GetNormalizedIntensity(P - (gGradientDeltaX), ch)) * gInvGradientDelta;
  Gradient.y = (GetNormalizedIntensity(P + (gGradientDeltaY), ch) - GetNormalizedIntensity(P - (gGradientDeltaY), ch)) * gInvGradientDelta;
  Gradient.z = (GetNormalizedIntensity(P + (gGradientDeltaZ), ch) - GetNormalizedIntensity(P - (gGradientDeltaZ), ch)) * gInvGradientDelta;

  return Gradient;
}


float GetOpacity(float NormalizedIntensity, int ch)
{
  // apply lut
  float Intensity = NormalizedIntensity;// * exp(1.0-1.0/g_opacity[ch]);
  return Intensity;
}

vec3 GetEmissionN(float NormalizedIntensity, int ch)
{
  return g_emissive[ch];
}

vec3 GetDiffuseN(float NormalizedIntensity, vec3 Pe, int ch)
{
  //return texture(g_colormapTexture[ch], vec2(0.5, 0.5)).xyz;

//  float i = NormalizedIntensity * (g_intensityMax[ch] - g_intensityMin[ch]) + g_intensityMin[ch];//(intensity - g_intensityMin) / (g_intensityMax - g_intensityMin)
//  i = (i-g_lutMin[ch])/(g_lutMax[ch]-g_lutMin[ch]) * g_opacity[ch];
//  return texture(g_colormapTexture[ch], vec2(i, 0.5)).xyz * g_diffuse[ch];

  vec4 intensity = UINT16_MAX * texture(volumeTexture, PtoVolumeTex(Pe));
  if (g_labels[ch] > 0.0) {
  return texelFetch(g_colormapTexture[ch], ivec2(int(intensity[ch]), 0), 0).xyz * g_diffuse[ch];
  }
  else {
  float i = intensity[ch];
  i = (i - g_lutMin[ch]) / (g_lutMax[ch] - g_lutMin[ch]);
  return texture(g_colormapTexture[ch], vec2(i, 0.5)).xyz * g_diffuse[ch];
  }


  //return g_diffuse[ch];
}

vec3 GetSpecularN(float NormalizedIntensity, int ch)
{
  return g_specular[ch];
}

float GetRoughnessN(float NormalizedIntensity, int ch)
{
  return g_roughness[ch];
}
)";

  const char* fsPiece3 = R"(
// a bsdf sample, a sample on a light source, and a randomly chosen light index
struct CLightingSample {
  float m_bsdfComponent;
  vec2  m_bsdfDir;
  vec2  m_lightPos;
  float m_lightComponent;
  float m_LightNum;
};

CLightingSample LightingSample_LargeStep(inout uvec2 seed) {
  return CLightingSample(
    rand(seed),
    vec2(rand(seed), rand(seed)),
    vec2(rand(seed), rand(seed)),
    rand(seed),
    rand(seed)
    );
}

// return a color xyz
vec3 Light_Le(in Light light, in vec2 UV)
{
  if (light.m_T == 0)
    return RGBtoXYZ(light.m_color) / light.m_area;

  if (light.m_T == 1)
  {
    if (UV.y > 0.0f)
      return RGBtoXYZ(mix(light.m_colorMiddle, light.m_colorTop, abs(UV.y)));
    else
      return RGBtoXYZ(mix(light.m_colorMiddle, light.m_colorBottom, abs(UV.y)));
  }

  return BLACK;
}

// return a color xyz
vec3 Light_SampleL(in Light light, in vec3 P, out Ray Rl, out float Pdf, in CLightingSample LS)
{
  vec3 L = BLACK;
  Pdf = 0.0;
  vec3 Ro = vec3(0,0,0), Rd = vec3(0,0,1);
  if (light.m_T == 0)
  {
    Ro = (light.m_P + ((-0.5f + LS.m_lightPos.x) * light.m_width * light.m_U) + ((-0.5f + LS.m_lightPos.y) * light.m_height * light.m_V));
    Rd = normalize(P - Ro);
    L = dot(Rd, light.m_N) > 0.0f ? Light_Le(light, vec2(0.0f)) : BLACK;
    Pdf = abs(dot(Rd, light.m_N)) > 0.0f ? dot(P-Ro, P-Ro) / (abs(dot(Rd, light.m_N)) * light.m_area) : 0.0f;
  }
  else if (light.m_T == 1)
  {
    Ro = light.m_P + light.m_skyRadius * getUniformSphereSample(LS.m_lightPos);
    Rd = normalize(P - Ro);
    L = Light_Le(light, vec2(1.0f) - 2.0f * LS.m_lightPos);
    Pdf = pow(light.m_skyRadius, 2.0f) / light.m_area;
  }

  Rl = Ray(Ro, Rd, 0.0f, length(P - Ro));

  return L;
}

// Intersect ray with light
bool Light_Intersect(Light light, inout Ray R, out float T, out vec3 L, out float pPdf)
{
  if (light.m_T == 0)
  {
    // Compute projection
    float DotN = dot(R.m_D, light.m_N);

    // Ray is coplanar with light surface
    if (DotN >= 0.0f)
      return false;

    // Compute hit distance
    T = (-light.m_distance - dot(R.m_O, light.m_N)) / DotN;

    // Intersection is in ray's negative direction
    if (T < R.m_MinT || T > R.m_MaxT)
      return false;

    // Determine position on light
    vec3 Pl = rayAt(R, T);

    // Vector from point on area light to center of area light
    vec3 Wl = Pl - light.m_P;

    // Compute texture coordinates
    vec2 UV = vec2(dot(Wl, light.m_U), dot(Wl, light.m_V));

    // Check if within bounds of light surface
    if (UV.x > light.m_halfWidth || UV.x < -light.m_halfWidth || UV.y > light.m_halfHeight || UV.y < -light.m_halfHeight)
      return false;

    R.m_MaxT = T;

    //pUV = UV;

    if (DotN < 0.0f)
      L = RGBtoXYZ(light.m_color) / light.m_area;
    else
      L = BLACK;

    pPdf = dot(R.m_O-Pl, R.m_O-Pl) / (DotN * light.m_area);

    return true;
  }

  else if (light.m_T == 1)
  {
    T = light.m_skyRadius;

    // Intersection is in ray's negative direction
    if (T < R.m_MinT || T > R.m_MaxT)
      return false;

    R.m_MaxT = T;

    vec2 UV = vec2(SphericalPhi(R.m_D) * INV_2_PI, SphericalTheta(R.m_D) * INV_PI);

    L = Light_Le(light, vec2(1.0f,1.0f) - 2.0f * UV);

    pPdf = pow(light.m_skyRadius, 2.0f) / light.m_area;
    //pUV = UV;

    return true;
  }

  return false;
}

float Light_Pdf(in Light light, in vec3 P, in vec3 Wi)
{
  vec3 L;
  vec2 UV;
  float Pdf = 1.0f;

  Ray Rl = Ray(P, Wi, 0.0f, 100000.0f);

  if (light.m_T == 0)
  {
    float T = 0.0f;

    if (!Light_Intersect(light, Rl, T, L, Pdf))
      return 0.0f;

    return pow(T, 2.0f) / (abs(dot(light.m_N, -Wi)) * light.m_area);
  }

  else if (light.m_T == 1)
  {
    return pow(light.m_skyRadius, 2.0f) / light.m_area;
  }

  return 0.0f;
}

struct CVolumeShader {
  int m_Type; // 0 = bsdf, 1 = phase

  vec3 m_Kd; // isotropic phase // xyz color
  vec3 m_R; // specular reflectance
  float m_Ior;
  float m_Exponent;
  vec3 m_Nn;
  vec3 m_Nu;
  vec3 m_Nv;
};

// return a xyz color
vec3 ShaderPhase_F(in CVolumeShader shader, in vec3 Wo, in vec3 Wi)
{
  return shader.m_Kd * INV_PI;
}

float ShaderPhase_Pdf(in CVolumeShader shader, in vec3 Wo, in vec3 Wi)
{
  return INV_4_PI;
}

vec3 ShaderPhase_SampleF(in CVolumeShader shader, in vec3 Wo, out vec3 Wi, out float Pdf, in vec2 U)
{
  Wi	= getUniformSphereSample(U);
  Pdf	= ShaderPhase_Pdf(shader, Wo, Wi);

  return ShaderPhase_F(shader, Wo, Wi);
}

// return a xyz color
vec3 Lambertian_F(in CVolumeShader shader, in vec3 Wo, in vec3 Wi)
{
  return shader.m_Kd * INV_PI;
}

float Lambertian_Pdf(in CVolumeShader shader, in vec3 Wo, in vec3 Wi)
{
  //return abs(Wi.z)*INV_PI;
  return SameHemisphere(Wo, Wi) ? abs(Wi.z) * INV_PI : 0.0f;
}

// return a xyz color
vec3 Lambertian_SampleF(in CVolumeShader shader, in vec3 Wo, out vec3 Wi, out float Pdf, in vec2 U)
{
  Wi = getCosineWeightedHemisphereSample(U);

  if (Wo.z < 0.0f)
    Wi.z *= -1.0f;

  Pdf = Lambertian_Pdf(shader, Wo, Wi);

  return Lambertian_F(shader, Wo, Wi);
}

vec3 SphericalDirection(in float SinTheta, in float CosTheta, in float Phi)
{
  return vec3(SinTheta * cos(Phi), SinTheta * sin(Phi), CosTheta);
}

void Blinn_SampleF(in CVolumeShader shader, in vec3 Wo, out vec3 Wi, out float Pdf, in vec2 U)
{
  // Compute sampled half-angle vector wh for Blinn distribution
  float costheta = pow(U.x, 1.f / (shader.m_Exponent+1.0));
  float sintheta = sqrt(max(0.f, 1.f - costheta*costheta));
  float phi = U.y * 2.f * PI;

  vec3 wh = SphericalDirection(sintheta, costheta, phi);

  if (!SameHemisphere(Wo, wh))
    wh = -wh;

  // Compute incident direction by reflecting about $\wh$
  Wi = -Wo + 2.f * dot(Wo, wh) * wh;

  // Compute PDF for wi from Blinn distribution
  float blinn_pdf = ((shader.m_Exponent + 1.f) * pow(costheta, shader.m_Exponent)) / (2.f * PI * 4.f * dot(Wo, wh));

  if (dot(Wo, wh) <= 0.f)
    blinn_pdf = 0.f;

  Pdf = blinn_pdf;
}

float Blinn_D(in CVolumeShader shader, in vec3 wh)
{
  float costhetah = abs(wh.z);//AbsCosTheta(wh);
  return (shader.m_Exponent+2.0) * INV_2_PI * pow(costhetah, shader.m_Exponent);
}
float Microfacet_G(in CVolumeShader shader, in vec3 wo, in vec3 wi, in vec3 wh)
{
  float NdotWh = abs(wh.z);//AbsCosTheta(wh);
  float NdotWo = abs(wo.z);//AbsCosTheta(wo);
  float NdotWi = abs(wi.z);//AbsCosTheta(wi);
  float WOdotWh = abs(dot(wo, wh));

  return min(1.f, min((2.f * NdotWh * NdotWo / WOdotWh), (2.f * NdotWh * NdotWi / WOdotWh)));
}

vec3 Microfacet_F(in CVolumeShader shader, in vec3 wo, in vec3 wi)
{
  float cosThetaO = abs(wo.z);//AbsCosTheta(wo);
  float cosThetaI = abs(wi.z);//AbsCosTheta(wi);

  if (cosThetaI == 0.f || cosThetaO == 0.f)
    return BLACK;

  vec3 wh = wi + wo;

  if (wh.x == 0. && wh.y == 0. && wh.z == 0.)
    return BLACK;

  wh = normalize(wh);
  float cosThetaH = dot(wi, wh);

  vec3 F = WHITE;//m_Fresnel.Evaluate(cosThetaH);

  return shader.m_R * Blinn_D(shader, wh) * Microfacet_G(shader, wo, wi, wh) * F / (4.f * cosThetaI * cosThetaO);
}

vec3 ShaderBsdf_WorldToLocal(in CVolumeShader shader, in vec3 W)
{
  return vec3(dot(W, shader.m_Nu), dot(W, shader.m_Nv), dot(W, shader.m_Nn));
}

vec3 ShaderBsdf_LocalToWorld(in CVolumeShader shader, in vec3 W)
{
  return vec3(	shader.m_Nu.x * W.x + shader.m_Nv.x * W.y + shader.m_Nn.x * W.z,
    shader.m_Nu.y * W.x + shader.m_Nv.y * W.y + shader.m_Nn.y * W.z,
    shader.m_Nu.z * W.x + shader.m_Nv.z * W.y + shader.m_Nn.z * W.z);
}

float Blinn_Pdf(in CVolumeShader shader, in vec3 Wo, in vec3 Wi)
{
  vec3 wh = normalize(Wo + Wi);

  float costheta = abs(wh.z);//AbsCosTheta(wh);
  // Compute PDF for wi from Blinn distribution
  float blinn_pdf = ((shader.m_Exponent + 1.f) * pow(costheta, shader.m_Exponent)) / (2.f * PI * 4.f * dot(Wo, wh));

  if (dot(Wo, wh) <= 0.0f)
    blinn_pdf = 0.0f;

  return blinn_pdf;
}

vec3 Microfacet_SampleF(in CVolumeShader shader, in vec3 wo, out vec3 wi, out float Pdf, in vec2 U)
{
  Blinn_SampleF(shader, wo, wi, Pdf, U);

  if (!SameHemisphere(wo, wi))
    return BLACK;

  return Microfacet_F(shader, wo, wi);
}

float Microfacet_Pdf(in CVolumeShader shader, in vec3 wo, in vec3 wi)
{
  if (!SameHemisphere(wo, wi))
    return 0.0f;

  return Blinn_Pdf(shader, wo, wi);
}

// return a xyz color
vec3 ShaderBsdf_F(in CVolumeShader shader, in vec3 Wo, in vec3 Wi)
{
  vec3 Wol = ShaderBsdf_WorldToLocal(shader, Wo);
  vec3 Wil = ShaderBsdf_WorldToLocal(shader, Wi);

  vec3 R = vec3(0,0,0);

  R += Lambertian_F(shader, Wol, Wil);
  R += Microfacet_F(shader, Wol, Wil);

  return R;
}

float ShaderBsdf_Pdf(in CVolumeShader shader, in vec3 Wo, in vec3 Wi)
{
  vec3 Wol = ShaderBsdf_WorldToLocal(shader, Wo);
  vec3 Wil = ShaderBsdf_WorldToLocal(shader, Wi);

  float Pdf = 0.0f;

  Pdf += Lambertian_Pdf(shader, Wol, Wil);
  Pdf += Microfacet_Pdf(shader, Wol, Wil);

  return Pdf;
}

vec3 ShaderBsdf_SampleF(in CVolumeShader shader, in CLightingSample S, in vec3 Wo, out vec3 Wi, out float Pdf, in vec2 U)
{
  vec3 Wol = ShaderBsdf_WorldToLocal(shader, Wo);
  vec3 Wil = vec3(0,0,0);

  vec3 R = vec3(0,0,0);

  if (S.m_bsdfComponent <= 0.5f)
  {
    Lambertian_SampleF(shader, Wol, Wil, Pdf, S.m_bsdfDir);
  }
  else
  {
    Microfacet_SampleF(shader, Wol, Wil, Pdf, S.m_bsdfDir);
  }

  Pdf += Lambertian_Pdf(shader, Wol, Wil);
  Pdf += Microfacet_Pdf(shader, Wol, Wil);

  R += Lambertian_F(shader, Wol, Wil);
  R += Microfacet_F(shader, Wol, Wil);

  Wi = ShaderBsdf_LocalToWorld(shader, Wil);

  //return vec3(1,1,1);
  return R;
}

// return a xyz color
vec3 Shader_F(in CVolumeShader shader, in vec3 Wo, in vec3 Wi)
{
  if (shader.m_Type == 0) {
    return ShaderBsdf_F(shader, Wo, Wi);
  }
  else {
    return ShaderPhase_F(shader, Wo, Wi);
  }
}

float Shader_Pdf(in CVolumeShader shader, in vec3 Wo, in vec3 Wi)
{
  if (shader.m_Type == 0) {
    return ShaderBsdf_Pdf(shader, Wo, Wi);
  }
  else {
    return ShaderPhase_Pdf(shader, Wo, Wi);
  }
}

vec3 Shader_SampleF(in CVolumeShader shader, in CLightingSample S, in vec3 Wo, out vec3 Wi, out float Pdf, in vec2 U)
{
  //return vec3(1,0,0);
  if (shader.m_Type == 0) {
    return ShaderBsdf_SampleF(shader, S, Wo, Wi, Pdf, U);
  }
  else {
    return ShaderPhase_SampleF(shader, Wo, Wi, Pdf, U);
  }
}
)";

  const char* fsPiece4 = R"(
bool IsBlack(in vec3 v) {
  return (v.x==0.0 && v.y == 0.0 && v.z == 0.0);
}

float PowerHeuristic(float nf, float fPdf, float ng, float gPdf)
{
  float f = nf * fPdf;
  float g = ng * gPdf;
  return (f * f) / (f * f + g * g);
}

// "shadow ray" using gStepSizeShadow, test whether it can exit the volume or not
bool FreePathRM(inout Ray R, inout uvec2 seed)
{
  float MinT;
  float MaxT;
  vec3 Ps;

  if (!IntersectBox(R, MinT, MaxT))
    return false;

  MinT = max(MinT, R.m_MinT);
  MaxT = min(MaxT, R.m_MaxT);

  float S	= -log(rand(seed)) / gDensityScale;
  float Sum = 0.0f;
  float SigmaT = 0.0f;

  MinT += rand(seed) * gStepSizeShadow;
  int ch = 0;
  float intensity = 0.0;
  while (Sum < S)
  {
    Ps = rayAt(R, MinT);  // R.m_O + MinT * R.m_D;

    if (MinT > MaxT)
      return false;

    //intensity = GetNormalizedIntensityRnd4ch_weighted(Ps, ch, seed);
    intensity = GetNormalizedIntensityMax4ch(Ps, ch);
    SigmaT = gDensityScale * GetOpacity(intensity, ch);

    Sum += SigmaT * gStepSizeShadow;
    MinT += gStepSizeShadow;
  }

  return true;
}


int NearestLight(Ray R, out vec3 LightColor, out vec3 Pl, out float oPdf)
{
  int Hit = -1;

  float T = 0.0f;

  Ray RayCopy = R;

  float Pdf = 0.0f;

  for (int i = 0; i < 2; i++)
  {
    if (Light_Intersect(gLights[i], RayCopy, T, LightColor, Pdf))
    {
      Pl = rayAt(R, T);
      Hit = i;
    }
  }

  oPdf = Pdf;

  return Hit;
}

// return a XYZ color
vec3 EstimateDirectLight(int shaderType, float Density, int ch, in Light light, in CLightingSample LS, in vec3 Wo, in vec3 Pe, in vec3 N, inout uvec2 seed)
{
  vec3 Ld = BLACK, Li = BLACK, F = BLACK;

  vec3 diffuse = GetDiffuseN(Density, Pe, ch);
  vec3 specular = GetSpecularN(Density, ch);
  float roughness = GetRoughnessN(Density, ch);

  vec3 nu = normalize(cross(N, Wo));
  vec3 nv = normalize(cross(N, nu));
  CVolumeShader Shader = CVolumeShader(shaderType, RGBtoXYZ(diffuse), RGBtoXYZ(specular), 2.5f, roughness, N, nu, nv);

  float LightPdf = 1.0f, ShaderPdf = 1.0f;


  Ray Rl = Ray(vec3(0,0,0), vec3(0,0,1.0), 0.0, 1500000.0f);
  Li = Light_SampleL(light, Pe, Rl, LightPdf, LS);

  vec3 Wi = -Rl.m_D, P = vec3(0,0,0);

  F = Shader_F(Shader,Wo, Wi);

  ShaderPdf = Shader_Pdf(Shader, Wo, Wi);

  if (!IsBlack(Li) && (ShaderPdf > 0.0f) && (LightPdf > 0.0f) && !FreePathRM(Rl, seed))
  {
    float WeightMIS = PowerHeuristic(1.0f, LightPdf, 1.0f, ShaderPdf);

    if (shaderType == ShaderType_Brdf){
      Ld += F * Li * abs(dot(Wi, N)) * WeightMIS / LightPdf;
    }

    else if (shaderType == ShaderType_Phase){
      Ld += F * Li * WeightMIS / LightPdf;
    }
  }

  F = Shader_SampleF(Shader, LS, Wo, Wi, ShaderPdf, LS.m_bsdfDir);

  if (!IsBlack(F) && (ShaderPdf > 0.0f))
  {
    vec3 Pl = vec3(0,0,0);
    int n = NearestLight(Ray(Pe, Wi, 0.0f, 1000000.0f), Li, Pl, LightPdf);
    if (n > -1)
    {
      Light pLight = gLights[n];
      LightPdf = Light_Pdf(pLight, Pe, Wi);

      if ((LightPdf > 0.0f) && !IsBlack(Li)) {
        Ray rr = Ray(Pl, normalize(Pe - Pl), 0.0f, length(Pe - Pl));
        if (!FreePathRM(rr, seed))
        {
          float WeightMIS = PowerHeuristic(1.0f, ShaderPdf, 1.0f, LightPdf);

          if (shaderType == ShaderType_Brdf) {
            Ld += F * Li * abs(dot(Wi, N)) * WeightMIS / ShaderPdf;

          }

          else if (shaderType == ShaderType_Phase) {
            Ld += F * Li * WeightMIS / ShaderPdf;
          }
        }

      }
    }
  }

  //return vec3(1,1,1);
  return Ld;
}

// return a linear xyz color
vec3 UniformSampleOneLight(int shaderType, float Density, int ch, in vec3 Wo, in vec3 Pe, in vec3 N, inout uvec2 seed)
{
  //if (NUM_LIGHTS == 0)
  //  return BLACK;

  // select a random light, a random 2d sample on light, and a random 2d sample on brdf
  CLightingSample LS = LightingSample_LargeStep(seed);

  int WhichLight = int(floor(LS.m_LightNum * float(NUM_LIGHTS)));

  Light light = gLights[WhichLight];

  return float(NUM_LIGHTS) * EstimateDirectLight(shaderType, Density, ch, light, LS, Wo, Pe, N, seed);

}

bool SampleDistanceRM(inout Ray R, inout uvec2 seed, out vec3 Ps, out float intensity, out int ch)
{
  float MinT;
  float MaxT;

  if (!IntersectBox(R, MinT, MaxT))
    return false;

  MinT = max(MinT, R.m_MinT);
  MaxT = min(MaxT, R.m_MaxT);

  // ray march along the ray's projected path and keep an average sigmaT value.
  // The distance is weighted by the intensity at each ray step sample. High intensity increases the apparent distance.
  // When the distance has become greater than the average sigmaT value given by -log(RandomFloat[0, 1]) / averageSigmaT
  // then that would be considered the interaction position.

  // sigmaT = sigmaA + sigmaS = absorption coeff + scattering coeff = extinction coeff

  // Beer-Lambert law: transmittance T(t) = exp(-sigmaT*t)
  // importance sampling the exponential function to produce a free path distance S
  // the PDF is p(t) = sigmaT * exp(-sigmaT * t)
  // S is the free-path distance = -ln(1-zeta)/sigmaT where zeta is a random variable
  float S	= -log(rand(seed)) / gDensityScale;  // note that ln(x:0..1) is negative

  // density scale 0... S --> 0..inf.  Low density means randomly sized ray paths
  // density scale inf... S --> 0.   High density means short ray paths!
  float Sum		= 0.0f;
  float SigmaT	= 0.0f; // accumulated extinction along ray march

  MinT += rand(seed) * gStepSize;
  //int ch = 0;
  //float intensity = 0.0;
  // ray march until we have traveled S (or hit the maxT of the ray)
  while (Sum < S)
  {
    Ps = rayAt(R, MinT);  // R.m_O + MinT * R.m_D;

    if (MinT > MaxT)
      return false;

    //intensity = GetNormalizedIntensityRnd4ch_weighted(Ps, ch, seed);
    intensity = GetNormalizedIntensityMax4ch(Ps, ch);
    SigmaT = gDensityScale * GetOpacity(intensity, ch);
    //SigmaT = gDensityScale * GetBlendedOpacity(volumedata, GetIntensity4ch(Ps, volumedata));

    Sum += SigmaT * gStepSize;
    MinT += gStepSize;
  }

  // Ps is the point
  return true;
}

uvec2 Sobol(uint n) {
    uvec2 p = uvec2(0u);
    uvec2 d = uvec2(0x80000000u);

    for(; n != 0u; n >>= 1u) {
        if((n & 1u) != 0u)
            p ^= d;

        d.x >>= 1u; // 1st dimension Sobol matrix, is same as base 2 Van der Corput
        d.y ^= d.y >> 1u; // 2nd dimension Sobol matrix
    }

    return p;
}

// adapted from: https://www.shadertoy.com/view/3lcczS
uint ReverseBits(uint x) {
    x = ((x & 0xaaaaaaaau) >> 1) | ((x & 0x55555555u) << 1);
    x = ((x & 0xccccccccu) >> 2) | ((x & 0x33333333u) << 2);
    x = ((x & 0xf0f0f0f0u) >> 4) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x & 0xff00ff00u) >> 8) | ((x & 0x00ff00ffu) << 8);
    return (x >> 16) | (x << 16);
    //return bitfieldReverse(x);
}

// EDIT: updated with a new hash that fixes an issue with the old one.
// details in the post linked at the top.
uint OwenHash(uint x, uint seed) { // works best with random seeds
    x ^= x * 0x3d20adeau;
    x += seed;
    x *= (seed >> 16) | 1u;
    x ^= x * 0x05526c56u;
    x ^= x * 0x53a22864u;
    return x;
}

uint OwenScramble(uint p, uint seed) {
    p = ReverseBits(p);
    p = OwenHash(p, seed);
    return ReverseBits(p);
}

vec2 OwenScrambledSobol(uint iter) {
    uvec2 ip = Sobol(iter);
    ip.x = OwenScramble(ip.x, 0xe7843fbfu);
    ip.y = OwenScramble(ip.y, 0x8d8fb1e0u);
    return vec2(ip) / float(0xffffffffu);
}

vec4 CalculateRadiance(inout uvec2 seed) {
  float r = rand(seed);
  //return vec4(r,0,0,1);

  vec3 Lv = BLACK, Li = BLACK;

  //Ray Re = Ray(vec3(0,0,0), vec3(0,0,1), 0.0, 1500000.0);
  //vec2 pixSample = vec2(rand(seed), rand(seed));
  vec2 pixSample = OwenScrambledSobol(uint(uSampleCounter));

  vec2 UV = vUv*uResolution + pixSample;

  Ray Re = GenerateCameraRay(gCamera, UV, vec2(rand(seed), rand(seed)));

  //return vec4(vUv, 0.0, 1.0);
  //return vec4(0.5*(Re.m_D + 1.0), 1.0);
  //return vec4(Re.m_D, 1.0);

  //Re.m_MinT = 0.0f;
  //Re.m_MaxT = 1500000.0f;

  vec3 Pe = vec3(0,0,0), Pl = vec3(0,0,0);
  float lpdf = 0.0;
  float alpha = 0.0;

  int ch;
  float D;
  // find point Pe along ray Re, and get its normalized intensity D and channel ch
  if (SampleDistanceRM(Re, seed, Pe, D, ch))
  {
    alpha = 1.0;
      //return vec4(1.0, 1.0, 1.0, 1.0);

    // is there a light between Re.m_O and Pe? (ray's maxT is distance to Pe)
    // (test to see if area light was hit before volume.)
    int i = NearestLight(Ray(Re.m_O, Re.m_D, 0.0f, length(Pe - Re.m_O)), Li, Pl, lpdf);
    if (i > -1)
    {
      // set sample pixel value in frame estimate (prior to accumulation)
      return vec4(Li, 1.0);
    }

    //int ch = 0;
    //float D = GetNormalizedIntensityMax4ch(Pe, ch);

    // emission from volume
    Lv += RGBtoXYZ(GetEmissionN(D, ch));

    vec3 gradient = Gradient4ch(Pe, ch);
    // send ray out from Pe toward light
    switch (gShadingType)
    {
      case 0:
      {
        Lv += UniformSampleOneLight(ShaderType_Brdf, D, ch, normalize(-Re.m_D), Pe, normalize(gradient), seed);
        break;
      }

      case 1:
      {
        Lv += 0.5f * UniformSampleOneLight(ShaderType_Phase, D, ch, normalize(-Re.m_D), Pe, normalize(gradient), seed);
        break;
      }

      case 2:
      {
        //const float GradMag = GradientMagnitude(Pe, volumedata.gradientVolumeTexture[ch]) * (1.0/volumedata.intensityMax[ch]);
        float GradMag = length(gradient);
        float PdfBrdf = (1.0f - exp(-gGradientFactor * GradMag));

        vec3 cls; // xyz color
        if (rand(seed) < PdfBrdf) {
          cls = UniformSampleOneLight(ShaderType_Brdf, D, ch, normalize(-Re.m_D), Pe, normalize(gradient), seed);
        }
        else {
          cls = 0.5f * UniformSampleOneLight(ShaderType_Phase, D, ch, normalize(-Re.m_D), Pe, normalize(gradient), seed);
        }

        Lv += cls;

        break;
      }
    }
  }
  else
  {
    // background color:
    // set Lv to a selected color based on environment light source?
//    if (uShowLights > 0.0) {
//      int n = NearestLight(Ray(Re.m_O, Re.m_D, 0.0f, 1000000.0f), Li, Pl, lpdf);
//      if (n > -1)
//        Lv = Li;
//    }

    //Lv = vec3(r,0,0);

  }

  // set sample pixel value in frame estimate (prior to accumulation)

  return vec4(Lv, alpha);
}

vec4 CumulativeMovingAverage(vec4 A, vec4 Ax, float N)
{
   return A + ((Ax - A) / max((N), 1.0f));
}

void main()
{
  // seed for rand(seed) function
  uvec2 seed = uvec2(uFrameCounter, uFrameCounter + 1.0) * uvec2(gl_FragCoord);

  // perform path tracing and get resulting pixel color
  vec4 pixelColor = CalculateRadiance( seed );

  vec4 previousColor = texture(tPreviousTexture, vUv);
  if (uSampleCounter < 1.0) {
    previousColor = vec4(0,0,0,0);
  }

  out_FragColor = CumulativeMovingAverage(previousColor, pixelColor, uSampleCounter);
}
)";

  m_fshader->compileSourceCode(
    (std::string(fsPiece1) + std::string(fsPiece2) + std::string(fsPiece3) + std::string(fsPiece4)).c_str());
  if (!m_fshader->isCompiled()) {
    LOG_ERROR << "GLPTVolumeShader: Failed to compile fragment shader\n" << m_fshader->log();
  }

  addShader(m_vshader);
  addShader(m_fshader);
  link();

  if (!isLinked()) {
    LOG_ERROR << "GLPTVolumeShader: Failed to link shader program\n" << log();
  }

  m_volumeTexture = uniformLocation("volumeTexture");

  m_tPreviousTexture = uniformLocation("tPreviousTexture");
  m_uSampleCounter = uniformLocation("uSampleCounter"); // 0
  m_uFrameCounter = uniformLocation("uFrameCounter");   // 1

  m_uResolution = uniformLocation("uResolution"); // : { type : "v2", value : new THREE.Vector2() },

  ///////////////////////////
  m_gClippedAaBbMin = uniformLocation("gClippedAaBbMin"); // : { type : "v3", value : new THREE.Vector3(0, 0, 0) },
  m_gClippedAaBbMax = uniformLocation("gClippedAaBbMax"); // : { type : "v3", value : new THREE.Vector3(1, 1, 1) },
  m_gDensityScale = uniformLocation("gDensityScale");     // : { type : "f", value : 50.0 },
  m_gStepSize = uniformLocation("gStepSize");             // : { type : "f", value : 1.0 },
  m_gStepSizeShadow = uniformLocation("gStepSizeShadow"); // : { type : "f", value : 1.0 },
  m_gPosToUVW = uniformLocation("gPosToUVW");             // : { type : "v3", value : new THREE.Vector3() },
  m_g_nChannels = uniformLocation("g_nChannels");         // : { type : "i", value : 1 },
  m_gShadingType = uniformLocation("gShadingType");       // : { type : "i", value : 2 },
  m_gGradientDeltaX = uniformLocation("gGradientDeltaX"); // : { type : "v3", value : new THREE.Vector3(0.01, 0, 0) },
  m_gGradientDeltaY = uniformLocation("gGradientDeltaY"); // : { type : "v3", value : new THREE.Vector3(0, 0.01, 0) },
  m_gGradientDeltaZ = uniformLocation("gGradientDeltaZ"); // : { type : "v3", value : new THREE.Vector3(0, 0, 0.01) },
  m_gInvGradientDelta = uniformLocation("gInvGradientDelta"); // : { type : "f", value : 0.0 },
  m_gGradientFactor = uniformLocation("gGradientFactor");     // : { type : "f", value : 0.5 },

  m_cameraFrom = uniformLocation("gCamera.m_from");
  m_cameraU = uniformLocation("gCamera.m_U");
  m_cameraV = uniformLocation("gCamera.m_V");
  m_cameraN = uniformLocation("gCamera.m_N");
  m_cameraScreen = uniformLocation("gCamera.m_screen");
  m_cameraInvScreen = uniformLocation("gCamera.m_invScreen");
  m_cameraFocalDistance = uniformLocation("gCamera.m_focalDistance");
  m_cameraApertureSize = uniformLocation("gCamera.m_apertureSize");
  m_cameraProjectionMode = uniformLocation("gCamera.m_isPerspective");

  // Camera struct
  //          m_from : new THREE.Vector3(),
  //          m_U : new THREE.Vector3(),
  //          m_V : new THREE.Vector3(),
  //          m_N : new THREE.Vector3(),
  //          m_screen : new THREE.Vector4(),    // left, right, bottom, top
  //          m_invScreen : new THREE.Vector2(), // 1/w, 1/h
  //          m_focalDistance : 0.0,
  //          m_apertureSize : 0.0

  m_light0theta = uniformLocation("gLights[0].m_theta");
  m_light0phi = uniformLocation("gLights[0].m_phi");
  m_light0width = uniformLocation("gLights[0].m_width");
  m_light0halfWidth = uniformLocation("gLights[0].m_halfWidth");
  m_light0height = uniformLocation("gLights[0].m_height");
  m_light0halfHeight = uniformLocation("gLights[0].m_halfHeight");
  m_light0distance = uniformLocation("gLights[0].m_distance");
  m_light0skyRadius = uniformLocation("gLights[0].m_skyRadius");
  m_light0P = uniformLocation("gLights[0].m_P");
  m_light0target = uniformLocation("gLights[0].m_target");
  m_light0N = uniformLocation("gLights[0].m_N");
  m_light0U = uniformLocation("gLights[0].m_U");
  m_light0V = uniformLocation("gLights[0].m_V");
  m_light0area = uniformLocation("gLights[0].m_area");
  m_light0areaPdf = uniformLocation("gLights[0].m_areaPdf");
  m_light0color = uniformLocation("gLights[0].m_color");
  m_light0colorTop = uniformLocation("gLights[0].m_colorTop");
  m_light0colorMiddle = uniformLocation("gLights[0].m_colorMiddle");
  m_light0colorBottom = uniformLocation("gLights[0].m_colorBottom");
  m_light0T = uniformLocation("gLights[0].m_T");

  m_light1theta = uniformLocation("gLights[1].m_theta");
  m_light1phi = uniformLocation("gLights[1].m_phi");
  m_light1width = uniformLocation("gLights[1].m_width");
  m_light1halfWidth = uniformLocation("gLights[1].m_halfWidth");
  m_light1height = uniformLocation("gLights[1].m_height");
  m_light1halfHeight = uniformLocation("gLights[1].m_halfHeight");
  m_light1distance = uniformLocation("gLights[1].m_distance");
  m_light1skyRadius = uniformLocation("gLights[1].m_skyRadius");
  m_light1P = uniformLocation("gLights[1].m_P");
  m_light1target = uniformLocation("gLights[1].m_target");
  m_light1N = uniformLocation("gLights[1].m_N");
  m_light1U = uniformLocation("gLights[1].m_U");
  m_light1V = uniformLocation("gLights[1].m_V");
  m_light1area = uniformLocation("gLights[1].m_area");
  m_light1areaPdf = uniformLocation("gLights[1].m_areaPdf");
  m_light1color = uniformLocation("gLights[1].m_color");
  m_light1colorTop = uniformLocation("gLights[1].m_colorTop");
  m_light1colorMiddle = uniformLocation("gLights[1].m_colorMiddle");
  m_light1colorBottom = uniformLocation("gLights[1].m_colorBottom");
  m_light1T = uniformLocation("gLights[1].m_T");

  // per channel

  m_lutTexture0 = uniformLocation("g_lutTexture[0]");
  m_lutTexture1 = uniformLocation("g_lutTexture[1]");
  m_lutTexture2 = uniformLocation("g_lutTexture[2]");
  m_lutTexture3 = uniformLocation("g_lutTexture[3]");
  m_colormapTexture0 = uniformLocation("g_colormapTexture[0]");
  m_colormapTexture1 = uniformLocation("g_colormapTexture[1]");
  m_colormapTexture2 = uniformLocation("g_colormapTexture[2]");
  m_colormapTexture3 = uniformLocation("g_colormapTexture[3]");
  m_intensityMax = uniformLocation("g_intensityMax");
  m_intensityMin = uniformLocation("g_intensityMin");
  m_lutMax = uniformLocation("g_lutMax");
  m_lutMin = uniformLocation("g_lutMin");
  m_labels = uniformLocation("g_labels");
  m_opacity = uniformLocation("g_opacity");
  m_emissive0 = uniformLocation("g_emissive[0]");
  m_emissive1 = uniformLocation("g_emissive[1]");
  m_emissive2 = uniformLocation("g_emissive[2]");
  m_emissive3 = uniformLocation("g_emissive[3]");
  m_diffuse0 = uniformLocation("g_diffuse[0]");
  m_diffuse1 = uniformLocation("g_diffuse[1]");
  m_diffuse2 = uniformLocation("g_diffuse[2]");
  m_diffuse3 = uniformLocation("g_diffuse[3]");
  m_specular0 = uniformLocation("g_specular[0]");
  m_specular1 = uniformLocation("g_specular[1]");
  m_specular2 = uniformLocation("g_specular[2]");
  m_specular3 = uniformLocation("g_specular[3]");
  m_roughness = uniformLocation("g_roughness");
  m_uShowLights = uniformLocation("uShowLights");
}

GLPTVolumeShader::~GLPTVolumeShader() {}

void
GLPTVolumeShader::setShadingUniforms(const Scene* scene,
                                     const DenoiseParams& denoise,
                                     const CCamera& cam,
                                     const CBoundingBox& clipped_bbox,
                                     const PathTraceRenderSettings& renderSettings,
                                     int numIterations,
                                     int randSeed,
                                     int w,
                                     int h,
                                     const ImageGpu& imggpu,
                                     GLuint accumulationTexture)
{
  check_gl("before pathtrace shader uniform binding");

  glUniform1i(m_volumeTexture, 0);
  glActiveTexture(GL_TEXTURE0 + 0);
  glBindTexture(GL_TEXTURE_2D, 0);
  glBindTexture(GL_TEXTURE_3D, imggpu.m_VolumeGLTexture);
  check_gl("post vol textures");

  glUniform1i(m_tPreviousTexture, 1);
  glActiveTexture(GL_TEXTURE0 + 1);
  glBindTexture(GL_TEXTURE_2D, accumulationTexture);
  check_gl("post accum textures");

  glUniform1f(m_uSampleCounter, (float)numIterations);
  glUniform1f(m_uFrameCounter, (float)(randSeed + 1));
  // glUniform1f(m_uFrameCounter, 1.0f);

  glUniform2f(m_uResolution, (float)w, (float)h);
  glUniform3fv(m_gClippedAaBbMax, 1, glm::value_ptr(clipped_bbox.GetMaxP()));
  glUniform3fv(m_gClippedAaBbMin, 1, glm::value_ptr(clipped_bbox.GetMinP()));

  ///////////////////////////
  glUniform1f(m_gDensityScale, renderSettings.m_DensityScale);
  glUniform1f(m_gStepSize, renderSettings.m_StepSizeFactor * renderSettings.m_GradientDelta);
  glUniform1f(m_gStepSizeShadow, renderSettings.m_StepSizeFactorShadow * renderSettings.m_GradientDelta);
  glUniform3fv(
    m_gPosToUVW,
    1,
    glm::value_ptr(scene->m_boundingBox.GetInverseExtent() * glm::vec3(scene->m_volume->getVolumeAxesFlipped())));

  glUniform1i(m_gShadingType, renderSettings.m_ShadingType);

  const float GradientDelta = 1.0f * renderSettings.m_GradientDelta;
  const float invGradientDelta = 1.0f / GradientDelta;
  const glm::vec3 GradientDeltaX(GradientDelta, 0.0f, 0.0f);
  const glm::vec3 GradientDeltaY(0.0f, GradientDelta, 0.0f);
  const glm::vec3 GradientDeltaZ(0.0f, 0.0f, GradientDelta);

  glUniform3fv(m_gGradientDeltaX, 1, glm::value_ptr(GradientDeltaX));
  glUniform3fv(m_gGradientDeltaY, 1, glm::value_ptr(GradientDeltaY));
  glUniform3fv(m_gGradientDeltaZ, 1, glm::value_ptr(GradientDeltaZ));
  glUniform1f(m_gInvGradientDelta, invGradientDelta);
  glUniform1f(m_gGradientFactor, renderSettings.m_GradientFactor);

  glUniform3fv(m_cameraFrom, 1, glm::value_ptr(cam.m_From));
  glUniform3fv(m_cameraU, 1, glm::value_ptr(cam.m_U));
  glUniform3fv(m_cameraV, 1, glm::value_ptr(cam.m_V));
  glUniform3fv(m_cameraN, 1, glm::value_ptr(cam.m_N));
  glUniform4fv(m_cameraScreen, 1, (float*)cam.m_Film.m_Screen);
  glUniform2fv(m_cameraInvScreen, 1, glm::value_ptr(cam.m_Film.m_InvScreen));
  glUniform1f(m_cameraFocalDistance, cam.m_Focus.m_FocalDistance);
  glUniform1f(m_cameraApertureSize, cam.m_Aperture.m_Size);
  glUniform1f(m_cameraProjectionMode, (cam.m_Projection == PERSPECTIVE) ? 1.0f : 0.0f);
  check_gl("pre lights");

  Light& l = scene->SphereLight();
  glUniform1f(m_light0theta, l.m_Theta);
  glUniform1f(m_light0phi, l.m_Phi);
  glUniform1f(m_light0width, l.m_Width);
  glUniform1f(m_light0halfWidth, l.m_HalfWidth);
  glUniform1f(m_light0height, l.m_Height);
  glUniform1f(m_light0halfHeight, l.m_HalfHeight);
  glUniform1f(m_light0distance, l.m_Distance);
  glUniform1f(m_light0skyRadius, l.m_SkyRadius);
  glUniform3fv(m_light0P, 1, glm::value_ptr(l.m_P));
  glUniform3fv(m_light0target, 1, glm::value_ptr(l.m_Target));
  glUniform3fv(m_light0N, 1, glm::value_ptr(l.m_N));
  glUniform3fv(m_light0U, 1, glm::value_ptr(l.m_U));
  glUniform3fv(m_light0V, 1, glm::value_ptr(l.m_V));
  glUniform1f(m_light0area, l.m_Area);
  glUniform1f(m_light0areaPdf, l.m_AreaPdf);
  glUniform3fv(m_light0color, 1, glm::value_ptr(l.m_Color * l.m_ColorIntensity));
  glUniform3fv(m_light0colorTop, 1, glm::value_ptr(l.m_ColorTop * l.m_ColorTopIntensity));
  glUniform3fv(m_light0colorMiddle, 1, glm::value_ptr(l.m_ColorMiddle * l.m_ColorMiddleIntensity));
  glUniform3fv(m_light0colorBottom, 1, glm::value_ptr(l.m_ColorBottom * l.m_ColorBottomIntensity));
  glUniform1i(m_light0T, l.m_T);

  Light& l1 = scene->AreaLight();
  glUniform1f(m_light1theta, l1.m_Theta);
  glUniform1f(m_light1phi, l1.m_Phi);
  glUniform1f(m_light1width, l1.m_Width);
  glUniform1f(m_light1halfWidth, l1.m_HalfWidth);
  glUniform1f(m_light1height, l1.m_Height);
  glUniform1f(m_light1halfHeight, l1.m_HalfHeight);
  glUniform1f(m_light1distance, l1.m_Distance);
  glUniform1f(m_light1skyRadius, l1.m_SkyRadius);
  glUniform3fv(m_light1P, 1, glm::value_ptr(l1.m_P));
  glUniform3fv(m_light1target, 1, glm::value_ptr(l1.m_Target));
  glUniform3fv(m_light1N, 1, glm::value_ptr(l1.m_N));
  glUniform3fv(m_light1U, 1, glm::value_ptr(l1.m_U));
  glUniform3fv(m_light1V, 1, glm::value_ptr(l1.m_V));
  glUniform1f(m_light1area, l1.m_Area);
  glUniform1f(m_light1areaPdf, l1.m_AreaPdf);
  glUniform3fv(m_light1color, 1, glm::value_ptr(l1.m_Color * l1.m_ColorIntensity));
  glUniform3fv(m_light1colorTop, 1, glm::value_ptr(l1.m_ColorTop * l1.m_ColorTopIntensity));
  glUniform3fv(m_light1colorMiddle, 1, glm::value_ptr(l1.m_ColorMiddle * l1.m_ColorMiddleIntensity));
  glUniform3fv(m_light1colorBottom, 1, glm::value_ptr(l1.m_ColorBottom * l1.m_ColorBottomIntensity));
  glUniform1i(m_light1T, l1.m_T);

  // per channel
  int NC = scene->m_volume->sizeC();

  int activeChannel = 0;
  int luttex[4] = { 0, 0, 0, 0 };
  int colormaptex[4] = { 0, 0, 0, 0 };
  float intensitymax[4] = { 1, 1, 1, 1 };
  float intensitymin[4] = { 0, 0, 0, 0 };
  float lutmax[4] = { 1, 1, 1, 1 };
  float lutmin[4] = { 0, 0, 0, 0 };
  float labels[4] = { 0, 0, 0, 0 };
  float diffuse[3 * 4] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
  float specular[3 * 4] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  float emissive[3 * 4] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
  float roughness[4] = { 0, 0, 0, 0 };
  float opacity[4] = { 0, 0, 0, 0 };
  for (int i = 0; i < NC; ++i) {
    if (scene->m_material.m_enabled[i] && activeChannel < MAX_GL_CHANNELS) {
      luttex[activeChannel] = imggpu.m_channels[i].m_VolumeLutGLTexture;
      colormaptex[activeChannel] = imggpu.m_channels[i].m_VolumeColorMapGLTexture;
      intensitymax[activeChannel] = scene->m_volume->channel(i)->m_max;
      intensitymin[activeChannel] = scene->m_volume->channel(i)->m_min;
      diffuse[activeChannel * 3 + 0] = scene->m_material.m_diffuse[i * 3 + 0];
      diffuse[activeChannel * 3 + 1] = scene->m_material.m_diffuse[i * 3 + 1];
      diffuse[activeChannel * 3 + 2] = scene->m_material.m_diffuse[i * 3 + 2];
      specular[activeChannel * 3 + 0] = scene->m_material.m_specular[i * 3 + 0];
      specular[activeChannel * 3 + 1] = scene->m_material.m_specular[i * 3 + 1];
      specular[activeChannel * 3 + 2] = scene->m_material.m_specular[i * 3 + 2];
      emissive[activeChannel * 3 + 0] = scene->m_material.m_emissive[i * 3 + 0];
      emissive[activeChannel * 3 + 1] = scene->m_material.m_emissive[i * 3 + 1];
      emissive[activeChannel * 3 + 2] = scene->m_material.m_emissive[i * 3 + 2];
      roughness[activeChannel] = scene->m_material.m_roughness[i];
      opacity[activeChannel] = scene->m_material.m_opacity[i];

      // get a min/max from the gradient data if possible
      uint16_t imin16 = 0;
      uint16_t imax16 = 0;
      bool hasMinMax =
        scene->m_material.m_gradientData[i].getMinMax(scene->m_volume->channel(i)->m_histogram, &imin16, &imax16);
      lutmin[activeChannel] = hasMinMax ? imin16 : intensitymin[activeChannel];
      lutmax[activeChannel] = hasMinMax ? imax16 : intensitymax[activeChannel];
      labels[activeChannel] = scene->m_material.m_labels[i];
      activeChannel++;
    }
  }
  glUniform1i(m_g_nChannels, activeChannel);
  check_gl("pre lut textures");

  glUniform1i(m_lutTexture0, 2);
  glActiveTexture(GL_TEXTURE0 + 2);
  glBindTexture(GL_TEXTURE_2D, luttex[0]);
  check_gl("lut 0");

  glUniform1i(m_lutTexture1, 3);
  glActiveTexture(GL_TEXTURE0 + 3);
  glBindTexture(GL_TEXTURE_2D, luttex[1]);
  check_gl("lut 1");

  glUniform1i(m_lutTexture2, 4);
  glActiveTexture(GL_TEXTURE0 + 4);
  glBindTexture(GL_TEXTURE_2D, luttex[2]);
  check_gl("lut 2");

  glUniform1i(m_lutTexture3, 5);
  glActiveTexture(GL_TEXTURE0 + 5);
  glBindTexture(GL_TEXTURE_2D, luttex[3]);
  check_gl("lut 3");

  glUniform1i(m_colormapTexture0, 6);
  glActiveTexture(GL_TEXTURE0 + 6);
  glBindTexture(GL_TEXTURE_2D, colormaptex[0]);
  check_gl("colormap 0");

  glUniform1i(m_colormapTexture1, 7);
  glActiveTexture(GL_TEXTURE0 + 7);
  glBindTexture(GL_TEXTURE_2D, colormaptex[1]);
  check_gl("colormap 1");

  glUniform1i(m_colormapTexture2, 8);
  glActiveTexture(GL_TEXTURE0 + 8);
  glBindTexture(GL_TEXTURE_2D, colormaptex[2]);
  check_gl("colormap 2");

  glUniform1i(m_colormapTexture3, 9);
  glActiveTexture(GL_TEXTURE0 + 9);
  glBindTexture(GL_TEXTURE_2D, colormaptex[3]);
  check_gl("colormap 3");

  glUniform4fv(m_intensityMax, 1, intensitymax);
  glUniform4fv(m_intensityMin, 1, intensitymin);
  glUniform4fv(m_lutMax, 1, lutmax);
  glUniform4fv(m_lutMin, 1, lutmin);
  glUniform4fv(m_labels, 1, labels);

  glUniform1fv(m_opacity, 4, opacity);
  glUniform3fv(m_emissive0, 1, emissive + 0);
  glUniform3fv(m_emissive1, 1, emissive + 3);
  glUniform3fv(m_emissive2, 1, emissive + 6);
  glUniform3fv(m_emissive3, 1, emissive + 9);
  glUniform3fv(m_diffuse0, 1, diffuse + 0);
  glUniform3fv(m_diffuse1, 1, diffuse + 3);
  glUniform3fv(m_diffuse2, 1, diffuse + 6);
  glUniform3fv(m_diffuse3, 1, diffuse + 9);
  glUniform3fv(m_specular0, 1, specular + 0);
  glUniform3fv(m_specular1, 1, specular + 3);
  glUniform3fv(m_specular2, 1, specular + 6);
  glUniform3fv(m_specular3, 1, specular + 9);
  glUniform1fv(m_roughness, 4, roughness);

  glUniform1i(m_uShowLights, 0);

  check_gl("pathtrace shader uniform binding");
}

void
GLPTVolumeShader::setTransformUniforms(const CCamera& camera, const glm::mat4& modelMatrix)
{}
