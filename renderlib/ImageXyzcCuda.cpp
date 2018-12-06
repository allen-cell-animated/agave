#include "ImageXyzcCuda.h"

#include "CudaUtilities.h"
#include "ImageXYZC.h"
#include "Logging.h"

#include <QElapsedTimer>
#include <QtDebug>

void
ChannelCuda::allocGpu(ImageXYZC* img, int channel, bool do_volume, bool do_gradient_volume)
{
  cudaExtent volumeSize;
  volumeSize.width = img->sizeX();
  volumeSize.height = img->sizeY();
  volumeSize.depth = img->sizeZ();

  Channelu16* ch = img->channel(channel);

  // intensity buffer
  if (do_volume) {
    // assuming 16-bit data!
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned);

    // create 3D array
    HandleCudaError(cudaMalloc3DArray(&m_volumeArray, &channelDesc, volumeSize));

    m_gpuBytes += (channelDesc.x + channelDesc.y + channelDesc.z + channelDesc.w) / 8 * volumeSize.width *
                  volumeSize.height * volumeSize.depth;

    // copy data to 3D array
    cudaMemcpy3DParms copyParams = { 0 };
    copyParams.srcPtr =
      make_cudaPitchedPtr(ch->m_ptr, volumeSize.width * img->sizeOfElement(), volumeSize.width, volumeSize.height);
    copyParams.dstArray = m_volumeArray;
    copyParams.extent = volumeSize;
    copyParams.kind = cudaMemcpyHostToDevice;
    HandleCudaError(cudaMemcpy3D(&copyParams));

    // create a texture object
    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));
    texRes.resType = cudaResourceTypeArray;
    texRes.res.array.array = m_volumeArray;
    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));
    texDescr.normalizedCoords = 1;
    texDescr.filterMode = cudaFilterModeLinear;
    texDescr.addressMode[0] = cudaAddressModeClamp; // clamp
    texDescr.addressMode[1] = cudaAddressModeClamp;
    texDescr.addressMode[2] = cudaAddressModeClamp;
    texDescr.readMode = cudaReadModeNormalizedFloat;
    HandleCudaError(cudaCreateTextureObject(&m_volumeTexture, &texRes, &texDescr, NULL));
  }

  // gradient buffer

  if (do_gradient_volume) {
    // assuming 16-bit data!
    cudaChannelFormatDesc gradientChannelDesc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned);

    // create 3D array
    HandleCudaError(cudaMalloc3DArray(&m_volumeGradientArray, &gradientChannelDesc, volumeSize));
    m_gpuBytes += (gradientChannelDesc.x + gradientChannelDesc.y + gradientChannelDesc.z + gradientChannelDesc.w) / 8 *
                  volumeSize.width * volumeSize.height * volumeSize.depth;

    // copy data to 3D array
    cudaMemcpy3DParms gradientCopyParams = { 0 };
    gradientCopyParams.srcPtr = make_cudaPitchedPtr(
      ch->m_gradientMagnitudePtr, volumeSize.width * img->sizeOfElement(), volumeSize.width, volumeSize.height);
    gradientCopyParams.dstArray = m_volumeGradientArray;
    gradientCopyParams.extent = volumeSize;
    gradientCopyParams.kind = cudaMemcpyHostToDevice;
    HandleCudaError(cudaMemcpy3D(&gradientCopyParams));

    // create a texture object
    cudaResourceDesc gradientTexRes;
    memset(&gradientTexRes, 0, sizeof(cudaResourceDesc));
    gradientTexRes.resType = cudaResourceTypeArray;
    gradientTexRes.res.array.array = m_volumeGradientArray;
    cudaTextureDesc gradientTexDescr;
    memset(&gradientTexDescr, 0, sizeof(cudaTextureDesc));
    gradientTexDescr.normalizedCoords = 1;
    gradientTexDescr.filterMode = cudaFilterModeLinear;
    gradientTexDescr.addressMode[0] = cudaAddressModeClamp; // clamp
    gradientTexDescr.addressMode[1] = cudaAddressModeClamp;
    gradientTexDescr.addressMode[2] = cudaAddressModeClamp;
    gradientTexDescr.readMode = cudaReadModeNormalizedFloat;
    HandleCudaError(cudaCreateTextureObject(&m_volumeGradientTexture, &gradientTexRes, &gradientTexDescr, NULL));
  }

  // LUT buffer

  const int LUT_SIZE = 256;
  cudaChannelFormatDesc lutChannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  // create 1D array
  HandleCudaError(cudaMallocArray(&m_volumeLutArray, &lutChannelDesc, LUT_SIZE, 1));
  m_gpuBytes += (lutChannelDesc.x + lutChannelDesc.y + lutChannelDesc.z + lutChannelDesc.w) / 8 * LUT_SIZE;
  // copy data to 1D array
  HandleCudaError(cudaMemcpyToArray(m_volumeLutArray, 0, 0, ch->m_lut, LUT_SIZE * 4, cudaMemcpyHostToDevice));

  // create texture objects
  cudaResourceDesc lutTexRes;
  memset(&lutTexRes, 0, sizeof(cudaResourceDesc));
  lutTexRes.resType = cudaResourceTypeArray;
  lutTexRes.res.array.array = m_volumeLutArray;
  cudaTextureDesc lutTexDescr;
  memset(&lutTexDescr, 0, sizeof(cudaTextureDesc));
  lutTexDescr.normalizedCoords = 1;
  lutTexDescr.filterMode = cudaFilterModeLinear;
  lutTexDescr.addressMode[0] = cudaAddressModeClamp; // clamp
  lutTexDescr.addressMode[1] = cudaAddressModeClamp;
  lutTexDescr.addressMode[2] = cudaAddressModeClamp;
  lutTexDescr.readMode = cudaReadModeElementType; // direct read the (filtered) value
  HandleCudaError(cudaCreateTextureObject(&m_volumeLutTexture, &lutTexRes, &lutTexDescr, NULL));
}

void
ChannelCuda::deallocGpu()
{
  HandleCudaError(cudaDestroyTextureObject(m_volumeLutTexture));
  m_volumeLutTexture = 0;
  HandleCudaError(cudaDestroyTextureObject(m_volumeGradientTexture));
  m_volumeGradientTexture = 0;
  HandleCudaError(cudaDestroyTextureObject(m_volumeTexture));
  m_volumeTexture = 0;

  HandleCudaError(cudaFreeArray(m_volumeLutArray));
  m_volumeLutArray = nullptr;
  HandleCudaError(cudaFreeArray(m_volumeGradientArray));
  m_volumeGradientArray = nullptr;
  HandleCudaError(cudaFreeArray(m_volumeArray));
  m_volumeArray = nullptr;

  m_gpuBytes = 0;
}

void
ChannelCuda::updateLutGpu(int channel, ImageXYZC* img)
{
  static const int LUT_SIZE = 256;
  HandleCudaError(
    cudaMemcpyToArray(m_volumeLutArray, 0, 0, img->channel(channel)->m_lut, LUT_SIZE * 4, cudaMemcpyHostToDevice));
}

void
ImageCuda::allocGpu(ImageXYZC* img)
{
  deallocGpu();
  m_channels.clear();

  QElapsedTimer timer;
  timer.start();

  for (uint32_t i = 0; i < img->sizeC(); ++i) {
    ChannelCuda c;
    c.m_index = i;
    c.allocGpu(img, i);
    m_channels.push_back(c);
  }

  LOG_DEBUG << "allocGPU: Image to GPU in " << timer.elapsed() << "ms";
}

void
ImageCuda::createVolumeTexture4x16(ImageXYZC* img, cudaArray_t* deviceArray, cudaTextureObject_t* deviceTexture)
{
  // assuming 16-bit data!
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned);

  // create 3D array
  cudaExtent volumeSize;
  volumeSize.width = img->sizeX();
  volumeSize.height = img->sizeY();
  volumeSize.depth = img->sizeZ();
  HandleCudaError(cudaMalloc3DArray(deviceArray, &channelDesc, volumeSize));
  m_gpuBytes += (channelDesc.x + channelDesc.y + channelDesc.z + channelDesc.w) / 8 * volumeSize.width *
                volumeSize.height * volumeSize.depth;

  // create texture tied to array
  cudaResourceDesc texRes;
  memset(&texRes, 0, sizeof(cudaResourceDesc));
  texRes.resType = cudaResourceTypeArray;
  texRes.res.array.array = *deviceArray;
  cudaTextureDesc texDescr;
  memset(&texDescr, 0, sizeof(cudaTextureDesc));
  texDescr.normalizedCoords = 1;
  texDescr.filterMode = cudaFilterModeLinear;
  texDescr.addressMode[0] = cudaAddressModeClamp; // clamp
  texDescr.addressMode[1] = cudaAddressModeClamp;
  texDescr.addressMode[2] = cudaAddressModeClamp;
  texDescr.readMode = cudaReadModeNormalizedFloat;
  HandleCudaError(cudaCreateTextureObject(deviceTexture, &texRes, &texDescr, NULL));
}

void
ImageCuda::updateVolumeData4x16(ImageXYZC* img, int c0, int c1, int c2, int c3)
{
  QElapsedTimer timer;
  timer.start();

  const int N = 4;
  int ch[4] = { c0, c1, c2, c3 };
  // interleaved all channels.
  // first 4.
  size_t xyz = img->sizeX() * img->sizeY() * img->sizeZ();
  uint16_t* v = new uint16_t[xyz * N];

  for (uint32_t i = 0; i < xyz; ++i) {
    for (int j = 0; j < N; ++j) {
      v[N * (i) + j] = img->channel(ch[j])->m_ptr[(i)];
    }
  }

  LOG_DEBUG << "Prepared interleaved hostmem buffer: " << timer.elapsed() << "ms";
  timer.start();

  // copy data to 3D array
  cudaExtent volumeSize;
  volumeSize.width = img->sizeX();
  volumeSize.height = img->sizeY();
  volumeSize.depth = img->sizeZ();
  cudaMemcpy3DParms copyParams = { 0 };
  copyParams.srcPtr =
    make_cudaPitchedPtr(v, volumeSize.width * img->sizeOfElement() * N, volumeSize.width, volumeSize.height);
  copyParams.dstArray = m_volumeArrayInterleaved;
  copyParams.extent = volumeSize;
  copyParams.kind = cudaMemcpyHostToDevice;
  HandleCudaError(cudaMemcpy3D(&copyParams));

  LOG_DEBUG << "Copy volume to gpu: " << timer.elapsed() << "ms";

  delete[] v;
}

void
ImageCuda::allocGpuInterleaved(ImageXYZC* img)
{
  deallocGpu();
  m_channels.clear();

  QElapsedTimer timer;
  timer.start();

  createVolumeTexture4x16(img, &m_volumeArrayInterleaved, &m_volumeTextureInterleaved);
  uint32_t numChannels = img->sizeC();
  updateVolumeData4x16(
    img, 0, std::min(1u, numChannels - 1), std::min(2u, numChannels - 1), std::min(3u, numChannels - 1));

  for (uint32_t i = 0; i < numChannels; ++i) {
    ChannelCuda c;
    c.m_index = i;
    c.allocGpu(img, i, false, false);
    m_channels.push_back(c);

    m_gpuBytes += c.m_gpuBytes;
  }

  LOG_DEBUG << "allocGPUinterleaved: Image to GPU in " << timer.elapsed() << "ms";
  LOG_DEBUG << "allocGPUinterleaved: GPU bytes: " << m_gpuBytes;
}

void
ImageCuda::deallocGpu()
{
  for (size_t i = 0; i < m_channels.size(); ++i) {
    m_channels[i].deallocGpu();
  }
  HandleCudaError(cudaDestroyTextureObject(m_volumeTextureInterleaved));
  m_volumeTextureInterleaved = 0;
  HandleCudaError(cudaFreeArray(m_volumeArrayInterleaved));
  m_volumeArrayInterleaved = nullptr;

  m_gpuBytes = 0;
}

void
ImageCuda::updateLutGpu(int channel, ImageXYZC* img)
{
  m_channels[channel].updateLutGpu(channel, img);
}
