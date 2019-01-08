#pragma once

class CBoundingBox;
class CCamera;
class DenoiseParams;
class Timing;
class CVariance;
struct CudaCamera;
struct CudaLighting;
struct PathTraceRenderSettings;

struct cudaBoundingBox
{
  float3 m_min;
  float3 m_max;
};

struct cudaFB
{
  float* m_fb;
  float* m_fbaccum;
  unsigned int* m_randomSeeds1;
  unsigned int* m_randomSeeds2;
};

#define MAX_CUDA_CHANNELS 4
struct cudaVolume
{
  int m_nChannels;
  float m_intensityMax[MAX_CUDA_CHANNELS];
  float m_intensityMin[MAX_CUDA_CHANNELS];
  float m_diffuse[MAX_CUDA_CHANNELS * 3];
  float m_specular[MAX_CUDA_CHANNELS * 3];
  float m_emissive[MAX_CUDA_CHANNELS * 3];
  float m_roughness[MAX_CUDA_CHANNELS];
  float m_opacity[MAX_CUDA_CHANNELS];

  cudaTextureObject_t m_volumeTexture[MAX_CUDA_CHANNELS];
  cudaTextureObject_t m_lutTexture[MAX_CUDA_CHANNELS];

  cudaVolume(int n) { m_nChannels = n; }
  ~cudaVolume() {}
};

void
BindConstants(const CudaLighting& cudalt,
              const DenoiseParams& denoise,
              const CudaCamera& cudacam,
              const cudaBoundingBox& bbox,
              const cudaBoundingBox& clipped_bbox,
              const PathTraceRenderSettings& renderSettings,
              int numIterations,
              int w,
              int h,
              float gamma,
              float exposure);

void
ComputeFocusDistance(const cudaVolume& volumedata);

void
Render(const int& Type,
       int numExposures,
       int w,
       int h,
       cudaFB& framebuffers,
       const cudaVolume& volumedata,
       Timing& RenderImage,
       Timing& BlurImage,
       Timing& PostProcessImage,
       Timing& DenoiseImage,
       int& numIterations);

void
ToneMap(float* inbuf, cudaSurfaceObject_t surfaceObj, int w, int h);

void
Denoise(float* inbuf, cudaSurfaceObject_t surfaceObj, int w, int h, float lerpC);
