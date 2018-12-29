#pragma once

#include <glad/glad.h>

#include <vector>

#include <cuda_runtime.h>

class ImageXYZC;

struct ChannelCuda
{
  cudaArray_t m_volumeArray = nullptr;
  cudaArray_t m_volumeGradientArray = nullptr;
  cudaArray_t m_volumeLutArray = nullptr;

  cudaTextureObject_t m_volumeTexture = 0;
  cudaTextureObject_t m_volumeGradientTexture = 0;
  cudaTextureObject_t m_volumeLutTexture = 0;

  int m_index;
  size_t m_gpuBytes = 0;

  void allocGpu(ImageXYZC* img, int channel, bool do_volume = true, bool do_gradient_volume = true);
  void deallocGpu();
  void updateLutGpu(int channel, ImageXYZC* img);
};

struct ImageCuda
{
  std::vector<ChannelCuda> m_channels;
  cudaArray_t m_volumeArrayInterleaved = nullptr;
  cudaTextureObject_t m_volumeTextureInterleaved = 0;
  
  GLuint m_VolumeGLTexture = 0;

  size_t m_gpuBytes = 0;

  // no one is calling this right now.
  // puts each channel into its own gpu volume buffer
  void allocGpu(ImageXYZC* img);

  // put first 4 channels into gpu array
  void allocGpuInterleaved(ImageXYZC* img);

  void deallocGpu();

  void updateLutGpu(int channel, ImageXYZC* img);

  void createVolumeTexture4x16(ImageXYZC* img);
  void createVolumeTextureFusedRGBA8(ImageXYZC* img);

  // similar to allocGpuInterleaved, change which channels are in the gpu volume buffer.
  void updateVolumeData4x16(ImageXYZC* img, int c0, int c1, int c2, int c3);

  ~ImageCuda() { deallocGpu(); }
};
