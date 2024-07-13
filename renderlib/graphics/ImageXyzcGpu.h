#pragma once

#include <glad/glad.h>

#include <vector>

class ImageXYZC;

struct ChannelGpu
{
  GLuint m_VolumeLutGLTexture = 0;
  GLuint m_VolumeColorMapGLTexture = 0;

  int m_index;
  size_t m_gpuBytes = 0;

  void allocGpu(ImageXYZC* img, int channel);
  void deallocGpu();
  void updateLutGpu(int channel, ImageXYZC* img);
};

struct ImageGpu
{
  std::vector<ChannelGpu> m_channels;

  GLuint m_VolumeGLTexture = 0;

  size_t m_gpuBytes = 0;

  // put first 4 channels into gpu array
  void allocGpuInterleaved(ImageXYZC* img, uint32_t c0 = 0u, uint32_t c1 = 1u, uint32_t c2 = 2u, uint32_t c3 = 3u);

  void deallocGpu();

  void updateLutGpu(int channel, ImageXYZC* img);

  void createVolumeTexture4x16(ImageXYZC* img);
  void createVolumeTextureFusedRGBA8(ImageXYZC* img);

  // similar to allocGpuInterleaved, change which channels are in the gpu volume buffer.
  void updateVolumeData4x16(ImageXYZC* img, int c0, int c1, int c2, int c3);

  void setVolumeTextureFiltering(bool linear);

  ~ImageGpu() { deallocGpu(); }
};
