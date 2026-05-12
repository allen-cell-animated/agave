#include "ImageXyzcGpu.h"

#include "AppScene.h" // for VolumeDisplay!  REFACTOR
#include "ImageXYZC.h"
#include "Logging.h"
#include "threading.h"

#include "gl/Util.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>

static constexpr int LUT_SIZE = 256;

// Toggle between updateVolumeData4x16 (original) and updateVolumeData4x16_optimized.
// Set the env var AGAVE_VOLUME_OPTIMIZED to any value to enable the optimized path.
// Evaluated once at first call.
static bool
useOptimizedVolumeUpload()
{
  static const bool enabled = (std::getenv("AGAVE_VOLUME_OPTIMIZED") != nullptr);
  return enabled;
}

ChannelGpu::~ChannelGpu()
{
  deallocGpu();
  m_VolumeLutGLTexture = 0;
  m_gpuBytes = 0;
  LOG_DEBUG << "ChannelGpu destructor called.";
}

void
ChannelGpu::allocGpu(ImageXYZC* img, int channel)
{
  Channelu16* ch = img->channel(channel);

  // LUT buffer

  m_gpuBytes += static_cast<size_t>(32) / 8 * LUT_SIZE;

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glGenTextures(1, &m_VolumeLutGLTexture);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, m_VolumeLutGLTexture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

  glTexStorage2D(GL_TEXTURE_2D, 1, GL_R32F, LUT_SIZE, 1);
  // glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, LUT_SIZE, 1, 0, GL_RED, GL_FLOAT, ch->m_lut);

  glBindTexture(GL_TEXTURE_2D, 0);
  check_gl("volume lut texture creation");
}

void
ChannelGpu::deallocGpu()
{
  glDeleteTextures(1, &m_VolumeLutGLTexture);
  m_VolumeLutGLTexture = 0;

  m_gpuBytes = 0;
}

void
ChannelGpu::updateLutGpu(int channel, ImageXYZC* img)
{
  static const int LUT_SIZE = 256;
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_3D, 0);
  check_gl("unbind to update lut texture");

  glBindTexture(GL_TEXTURE_2D, m_VolumeLutGLTexture);
  check_gl("bind lut texture");
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, LUT_SIZE, 1, GL_RED, GL_FLOAT, img->channel(channel)->m_lut);
  check_gl("update lut texture");

  glBindTexture(GL_TEXTURE_2D, 0);
  check_gl("unbind lut texture");
}

void
ImageGpu::createVolumeTextureFusedRGBA8(ImageXYZC* img)
{
  m_gpuBytes += (8 + 8 + 8 + 8) / 8 * img->sizeX() * img->sizeY() * img->sizeZ();

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glGenTextures(1, &m_VolumeGLTexture);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, 0);
  glBindTexture(GL_TEXTURE_3D, m_VolumeGLTexture);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  glTexStorage3D(GL_TEXTURE_3D, 1, GL_RGBA8, img->sizeX(), img->sizeY(), img->sizeZ());
  glBindTexture(GL_TEXTURE_3D, 0);
  check_gl("fused rgb volume texture creation");
}

void
ImageGpu::createVolumeTexture4x16(ImageXYZC* img)
{
  int N = 4;
  if (img->sizeC() < 4) {
    N = img->sizeC();
  }

  m_gpuBytes += (16 * N) / 8 * img->sizeX() * img->sizeY() * img->sizeZ();

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glGenTextures(1, &m_VolumeGLTexture);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, 0);
  glBindTexture(GL_TEXTURE_3D, m_VolumeGLTexture);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);

  GLenum internalFormat = GL_RGBA16;
  if (img->sizeC() == 3) {
    internalFormat = GL_RGB16;
  } else if (img->sizeC() == 2) {
    internalFormat = GL_RG16;
  } else if (img->sizeC() == 1) {
    internalFormat = GL_R16;
  }
  glTexStorage3D(GL_TEXTURE_3D, 1, internalFormat, img->sizeX(), img->sizeY(), img->sizeZ());
  glBindTexture(GL_TEXTURE_3D, 0);

  glGenTextures(1, &m_ActiveChannelColormaps);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D_ARRAY, m_ActiveChannelColormaps);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGBA8, LUT_SIZE, 1, 4);
  // 4 * glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, LUT_SIZE, 1);
  glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

  check_gl("volume texture creation");
}

void
ImageGpu::updateVolumeData4x16(ImageXYZC* img, int c0, int c1, int c2, int c3)
{
  auto startTime = std::chrono::high_resolution_clock::now();

  size_t N = 4;
  if (img->sizeC() < 4) {
    N = img->sizeC();
  }
  int ch[4] = { c0, c1, c2, c3 };
  // interleaved all channels.
  // up to the first 4.
  size_t xyz = img->sizeX() * img->sizeY() * img->sizeZ();
  uint16_t* v = new uint16_t[xyz * N];

  parallel_for(xyz, [&N, &v, &img, &ch](size_t s, size_t e) {
    for (size_t i = s; i < e; ++i) {
      for (size_t j = 0; j < N; ++j) {
        v[N * (i) + j] = img->channel(ch[j])->m_ptr[(i)];
      }
    }
  });

  auto endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = endTime - startTime;
  LOG_DEBUG << "Prepared interleaved hostmem buffer: " << (elapsed.count() * 1000.0) << "ms";

  startTime = std::chrono::high_resolution_clock::now();

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, 0);
  glBindTexture(GL_TEXTURE_3D, m_VolumeGLTexture);
  GLenum dataFormat = GL_RGBA;
  if (img->sizeC() == 1) {
    dataFormat = GL_RED;
  } else if (img->sizeC() == 2) {
    dataFormat = GL_RG;
  } else if (img->sizeC() == 3) {
    dataFormat = GL_RGB;
  }
  try {
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    // do this in chunks for very large data sizes.
    static const size_t GB = static_cast<size_t>(1024) * 1024 * 1024;
    size_t chunkThresholdBytes = 4 * GB;
    // if whole volume is larger than chunk size:
    if (xyz * N * sizeof(uint16_t) > chunkThresholdBytes) {
      // split the operation into chunks.
      // find number of z planes that fit into 1 chunk:
      size_t zPlanes = img->sizeZ();
      size_t planeSizeElements = img->sizeX() * img->sizeY() * N;
      size_t zPlanesPerChunk = chunkThresholdBytes / (planeSizeElements * sizeof(uint16_t));
      size_t numChunks = (zPlanes + zPlanesPerChunk - 1) / zPlanesPerChunk;
      LOG_DEBUG << "Updating volume texture in " << numChunks << " chunks of " << zPlanesPerChunk
                << " z-planes each, total size: " << (xyz * sizeof(uint16_t) * N) << " bytes";
      // loop over all the chunks and call glTexSubImage3D with the right z offset for each
      for (size_t chunk = 0, zoffset = 0; chunk < numChunks; ++chunk) {
        // only the last chunk could have less zplanes.
        size_t chunkSizeZ = std::min(zPlanes - zoffset, zPlanesPerChunk);
        glTexSubImage3D(GL_TEXTURE_3D,
                        0,
                        0,
                        0,
                        zoffset,
                        img->sizeX(),
                        img->sizeY(),
                        chunkSizeZ,
                        dataFormat,
                        GL_UNSIGNED_SHORT,
                        v + zoffset * planeSizeElements);
        zoffset += chunkSizeZ;
      }
    } else {
      glTexSubImage3D(
        GL_TEXTURE_3D, 0, 0, 0, 0, img->sizeX(), img->sizeY(), img->sizeZ(), dataFormat, GL_UNSIGNED_SHORT, v);
    }

  } catch (const std::exception& e) {
    LOG_ERROR << "Failed to update volume texture (" << img->sizeX() << ", " << img->sizeY() << ", " << img->sizeZ()
              << "): " << e.what();
    delete[] v;
    return;
  } catch (...) {
    LOG_ERROR << "Failed to update volume texture (" << img->sizeX() << ", " << img->sizeY() << ", " << img->sizeZ()
              << "): unknown error";
    delete[] v;
    return;
  }
  glBindTexture(GL_TEXTURE_3D, 0);
  check_gl("update volume texture");

  endTime = std::chrono::high_resolution_clock::now();
  elapsed = endTime - startTime;
  LOG_DEBUG << "Copy volume to gpu: " << (xyz * sizeof(uint16_t) * N) << " bytes in " << (elapsed.count() * 1000.0)
            << "ms";

  delete[] v;
}

namespace {

// Specialized interleave: src[0..N-1] -> dst with stride N. N is a compile-time template
// parameter so the inner loop is a fixed-size store the compiler can vectorize.
template<size_t N>
void
interleaveChannels(uint16_t* __restrict dst, const uint16_t* const* __restrict src, size_t start, size_t end)
{
  if constexpr (N == 1) {
    // Pure copy; caller should normally take the no-copy fast path instead.
    std::memcpy(dst + start, src[0] + start, (end - start) * sizeof(uint16_t));
  } else {
    const uint16_t* s0 = src[0];
    const uint16_t* s1 = src[1];
    const uint16_t* s2 = (N > 2) ? src[2] : nullptr;
    const uint16_t* s3 = (N > 3) ? src[3] : nullptr;
    for (size_t i = start; i < end; ++i) {
      uint16_t* d = dst + N * i;
      d[0] = s0[i];
      d[1] = s1[i];
      if constexpr (N > 2)
        d[2] = s2[i];
      if constexpr (N > 3)
        d[3] = s3[i];
    }
  }
}

void
runInterleave(size_t N, uint16_t* dst, const uint16_t* const* src, size_t xyz)
{
  switch (N) {
    case 1:
      parallel_for(xyz, [dst, src](size_t s, size_t e) { interleaveChannels<1>(dst, src, s, e); });
      break;
    case 2:
      parallel_for(xyz, [dst, src](size_t s, size_t e) { interleaveChannels<2>(dst, src, s, e); });
      break;
    case 3:
      parallel_for(xyz, [dst, src](size_t s, size_t e) { interleaveChannels<3>(dst, src, s, e); });
      break;
    case 4:
    default:
      parallel_for(xyz, [dst, src](size_t s, size_t e) { interleaveChannels<4>(dst, src, s, e); });
      break;
  }
}

} // namespace

void
ImageGpu::updateVolumeData4x16_optimized(ImageXYZC* img, int c0, int c1, int c2, int c3)
{
  auto startTime = std::chrono::high_resolution_clock::now();

  size_t N = 4;
  if (img->sizeC() < 4) {
    N = img->sizeC();
  }
  const int ch[4] = { c0, c1, c2, c3 };
  const size_t xyz = static_cast<size_t>(img->sizeX()) * img->sizeY() * img->sizeZ();
  const size_t totalBytes = xyz * N * sizeof(uint16_t);

  // Hoist channel base pointers out of the inner loop.
  const uint16_t* src[4] = { nullptr, nullptr, nullptr, nullptr };
  for (size_t j = 0; j < N; ++j) {
    src[j] = img->channel(ch[j])->m_ptr;
  }

  GLenum dataFormat = GL_RGBA;
  if (img->sizeC() == 1) {
    dataFormat = GL_RED;
  } else if (img->sizeC() == 2) {
    dataFormat = GL_RG;
  } else if (img->sizeC() == 3) {
    dataFormat = GL_RGB;
  }

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, 0);
  glBindTexture(GL_TEXTURE_3D, m_VolumeGLTexture);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  // ---- Fast path: N==1, no interleave needed, upload straight from source. ----
  if (N == 1) {
    auto endPrep = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> prepElapsed = endPrep - startTime;
    LOG_DEBUG << "Prepared interleaved hostmem buffer: " << (prepElapsed.count() * 1000.0)
              << "ms (optimized, N==1 no-copy)";

    auto copyStart = std::chrono::high_resolution_clock::now();

    static const size_t GB = static_cast<size_t>(1024) * 1024 * 1024;
    const size_t chunkThresholdBytes = 4 * GB;
    if (totalBytes > chunkThresholdBytes) {
      const size_t zPlanes = img->sizeZ();
      const size_t planeSizeElements = static_cast<size_t>(img->sizeX()) * img->sizeY();
      const size_t zPlanesPerChunk = chunkThresholdBytes / (planeSizeElements * sizeof(uint16_t));
      const size_t numChunks = (zPlanes + zPlanesPerChunk - 1) / zPlanesPerChunk;
      LOG_DEBUG << "Updating volume texture in " << numChunks << " chunks of " << zPlanesPerChunk << " z-planes each";
      for (size_t chunk = 0, zoffset = 0; chunk < numChunks; ++chunk) {
        const size_t chunkSizeZ = std::min(zPlanes - zoffset, zPlanesPerChunk);
        glTexSubImage3D(GL_TEXTURE_3D,
                        0,
                        0,
                        0,
                        static_cast<GLint>(zoffset),
                        img->sizeX(),
                        img->sizeY(),
                        static_cast<GLsizei>(chunkSizeZ),
                        dataFormat,
                        GL_UNSIGNED_SHORT,
                        src[0] + zoffset * planeSizeElements);
        zoffset += chunkSizeZ;
      }
    } else {
      glTexSubImage3D(
        GL_TEXTURE_3D, 0, 0, 0, 0, img->sizeX(), img->sizeY(), img->sizeZ(), dataFormat, GL_UNSIGNED_SHORT, src[0]);
    }
    glBindTexture(GL_TEXTURE_3D, 0);
    check_gl("update volume texture (optimized N==1)");

    auto copyEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> copyElapsed = copyEnd - copyStart;
    LOG_DEBUG << "Copy volume to gpu: " << totalBytes << " bytes in " << (copyElapsed.count() * 1000.0)
              << "ms (optimized, N==1)";
    return;
  }

  // ---- N>=2 path: interleave directly into a mapped PBO, then DMA upload. ----
  // Chunking strategy mirrors the original: keep each upload <= 4GB.
  static const size_t GB = static_cast<size_t>(1024) * 1024 * 1024;
  const size_t chunkThresholdBytes = 4 * GB;
  const size_t planeSizeElements = static_cast<size_t>(img->sizeX()) * img->sizeY() * N;
  const size_t planeBytes = planeSizeElements * sizeof(uint16_t);

  size_t zPlanesPerChunk = img->sizeZ();
  if (totalBytes > chunkThresholdBytes) {
    zPlanesPerChunk = chunkThresholdBytes / planeBytes;
    if (zPlanesPerChunk == 0) {
      zPlanesPerChunk = 1;
    }
  }
  const size_t chunkBytesMax = zPlanesPerChunk * planeBytes;

  GLuint pbo = 0;
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
  glBufferData(GL_PIXEL_UNPACK_BUFFER, static_cast<GLsizeiptr>(chunkBytesMax), nullptr, GL_STREAM_DRAW);
  check_gl("optimized volume PBO alloc");

  // Heap fallback if mapping ever fails (kept simple; not currently expected).
  std::unique_ptr<uint16_t[]> heapFallback;

  double prepMsTotal = 0.0;
  double copyMsTotal = 0.0;

  const size_t zPlanes = img->sizeZ();
  size_t numChunks = (zPlanes + zPlanesPerChunk - 1) / zPlanesPerChunk;
  if (numChunks > 1) {
    LOG_DEBUG << "Updating volume texture in " << numChunks << " chunks of " << zPlanesPerChunk
              << " z-planes each, total size: " << totalBytes << " bytes";
  }

  for (size_t chunk = 0, zoffset = 0; chunk < numChunks; ++chunk) {
    const size_t chunkSizeZ = std::min(zPlanes - zoffset, zPlanesPerChunk);
    const size_t chunkElements = chunkSizeZ * static_cast<size_t>(img->sizeX()) * img->sizeY();
    const size_t chunkBytes = chunkElements * N * sizeof(uint16_t);
    const size_t srcOffsetElements = zoffset * static_cast<size_t>(img->sizeX()) * img->sizeY();

    auto prepStart = std::chrono::high_resolution_clock::now();

    void* mapped = glMapBufferRange(GL_PIXEL_UNPACK_BUFFER,
                                    0,
                                    static_cast<GLsizeiptr>(chunkBytes),
                                    GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT);
    check_gl("optimized volume PBO map");

    uint16_t* dst = nullptr;
    if (mapped) {
      dst = static_cast<uint16_t*>(mapped);
    } else {
      // Fallback: write into a heap buffer and use glBufferSubData.
      LOG_WARNING << "glMapBufferRange returned null; falling back to heap buffer for volume upload";
      heapFallback.reset(new uint16_t[chunkElements * N]);
      dst = heapFallback.get();
    }

    // Build per-chunk source pointers offset to the current z slab.
    const uint16_t* chunkSrc[4] = { nullptr, nullptr, nullptr, nullptr };
    for (size_t j = 0; j < N; ++j) {
      chunkSrc[j] = src[j] + srcOffsetElements;
    }
    runInterleave(N, dst, chunkSrc, chunkElements);

    if (mapped) {
      glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
      check_gl("optimized volume PBO unmap");
    } else {
      glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, static_cast<GLsizeiptr>(chunkBytes), heapFallback.get());
      check_gl("optimized volume PBO heap fallback upload");
    }

    auto prepEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> prepElapsed = prepEnd - prepStart;
    prepMsTotal += prepElapsed.count() * 1000.0;

    auto copyStart = std::chrono::high_resolution_clock::now();
    glTexSubImage3D(GL_TEXTURE_3D,
                    0,
                    0,
                    0,
                    static_cast<GLint>(zoffset),
                    img->sizeX(),
                    img->sizeY(),
                    static_cast<GLsizei>(chunkSizeZ),
                    dataFormat,
                    GL_UNSIGNED_SHORT,
                    nullptr); // sourced from bound PBO
    check_gl("optimized volume texSubImage3D");
    auto copyEnd = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> copyElapsed = copyEnd - copyStart;
    copyMsTotal += copyElapsed.count() * 1000.0;

    zoffset += chunkSizeZ;
  }

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
  glDeleteBuffers(1, &pbo);
  glBindTexture(GL_TEXTURE_3D, 0);

  LOG_DEBUG << "Prepared interleaved hostmem buffer: " << prepMsTotal << "ms (optimized, N=" << N << ", PBO)";
  LOG_DEBUG << "Copy volume to gpu: " << totalBytes << " bytes in " << copyMsTotal << "ms (optimized, N=" << N
            << ", PBO)";
}

void
ImageGpu::allocGpuInterleaved(ImageXYZC* img, uint32_t c0, uint32_t c1, uint32_t c2, uint32_t c3)
{
  deallocGpu();

  auto startTime = std::chrono::high_resolution_clock::now();

  createVolumeTexture4x16(img);
  uint32_t numChannels = img->sizeC();
  const int cc0 = static_cast<int>(std::min(c0, numChannels - 1));
  const int cc1 = static_cast<int>(std::min(c1, numChannels - 1));
  const int cc2 = static_cast<int>(std::min(c2, numChannels - 1));
  const int cc3 = static_cast<int>(std::min(c3, numChannels - 1));
  if (useOptimizedVolumeUpload()) {
    LOG_DEBUG << "allocGPUinterleaved: using OPTIMIZED volume upload path";
    updateVolumeData4x16_optimized(img, cc0, cc1, cc2, cc3);
  } else {
    updateVolumeData4x16(img, cc0, cc1, cc2, cc3);
  }

  for (uint32_t i = 0; i < numChannels; ++i) {
    ChannelGpu* c = new ChannelGpu(i);
    c->allocGpu(img, i);
    m_channels.push_back(c);

    m_gpuBytes += c->m_gpuBytes;
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = endTime - startTime;
  LOG_DEBUG << "allocGPUinterleaved: Image to GPU in " << (elapsed.count() * 1000.0) << "ms";
  LOG_DEBUG << "allocGPUinterleaved: GPU bytes: " << m_gpuBytes;
}

void
ImageGpu::deallocGpu()
{
  for (size_t i = 0; i < m_channels.size(); ++i) {
    delete m_channels[i];
  }
  m_channels.clear();

  // needs current gl context.

  check_gl("pre-destroy gl volume texture");
  //  glBindTexture(GL_TEXTURE_3D, 0);
  glDeleteTextures(1, &m_VolumeGLTexture);
  check_gl("destroy gl volume texture");
  LOG_DEBUG << "deallocGPU: GPU bytes: " << m_gpuBytes;
  m_VolumeGLTexture = 0;

  glDeleteTextures(1, &m_ActiveChannelColormaps);
  check_gl("destroy gl colormaps texture");
  m_ActiveChannelColormaps = 0;

  m_gpuBytes = 0;

  glFinish();
}

void
ImageGpu::updateLutGPU(ImageXYZC* img, int c0, int c1, int c2, int c3, const VolumeDisplay& volumeDisplay)
{
  // and write color luts
  glBindTexture(GL_TEXTURE_2D_ARRAY, m_ActiveChannelColormaps);
  // mip level, xy offset, layer index
  glTexSubImage3D(GL_TEXTURE_2D_ARRAY,
                  0,
                  0,
                  0,
                  0,
                  LUT_SIZE,
                  1,
                  1,
                  GL_RGBA,
                  GL_UNSIGNED_BYTE,
                  volumeDisplay.m_colormap[c0].m_colormap.data());
  glTexSubImage3D(GL_TEXTURE_2D_ARRAY,
                  0,
                  0,
                  0,
                  1,
                  LUT_SIZE,
                  1,
                  1,
                  GL_RGBA,
                  GL_UNSIGNED_BYTE,
                  volumeDisplay.m_colormap[c1].m_colormap.data());
  glTexSubImage3D(GL_TEXTURE_2D_ARRAY,
                  0,
                  0,
                  0,
                  2,
                  LUT_SIZE,
                  1,
                  1,
                  GL_RGBA,
                  GL_UNSIGNED_BYTE,
                  volumeDisplay.m_colormap[c2].m_colormap.data());
  glTexSubImage3D(GL_TEXTURE_2D_ARRAY,
                  0,
                  0,
                  0,
                  3,
                  LUT_SIZE,
                  1,
                  1,
                  GL_RGBA,
                  GL_UNSIGNED_BYTE,
                  volumeDisplay.m_colormap[c3].m_colormap.data());
  glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
}

void
ImageGpu::updateLutGpu(int channel, ImageXYZC* img)
{
  m_channels[channel]->updateLutGpu(channel, img);
}

void
ImageGpu::setVolumeTextureFiltering(bool linear)
{
  glBindTexture(GL_TEXTURE_3D, m_VolumeGLTexture);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, linear ? GL_LINEAR : GL_NEAREST);
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, linear ? GL_LINEAR : GL_NEAREST);
  glBindTexture(GL_TEXTURE_3D, 0);
}

ImageGpu::~ImageGpu()
{
  deallocGpu();
  LOG_DEBUG << "ImageGpu destructor called.";
}
