#pragma once

#include <glad/glad.h>

#include <vector>

class ImageXYZC;

struct ChannelCuda
{
  GLuint m_VolumeLutGLTexture = 0;

  int m_index;
  size_t m_gpuBytes = 0;

  void allocGpu(ImageXYZC* img, int channel);
  void deallocGpu();
  void updateLutGpu(int channel, ImageXYZC* img);
};

struct ImageCuda
{
  std::vector<ChannelCuda> m_channels;
  
  GLuint m_VolumeGLTexture = 0;

  size_t m_gpuBytes = 0;

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
