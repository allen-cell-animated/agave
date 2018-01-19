#pragma once

#include <vector>

#include <cuda_runtime.h>

class ImageXYZC;

struct ChannelCuda {
	cudaArray_t _volumeArray = nullptr;
	cudaArray_t _volumeGradientArray = nullptr;
	cudaArray_t _volumeLutArray = nullptr;

    cudaTextureObject_t _volumeTexture = 0;
    cudaTextureObject_t _volumeGradientTexture = 0;
    cudaTextureObject_t _volumeLutTexture = 0;

    int _index;

    void allocGpu(ImageXYZC* img, int channel, bool do_volume = true);
    void deallocGpu();
	void updateLutGpu(int channel, ImageXYZC* img);

};

struct ImageCuda {
    std::vector<ChannelCuda> _channels;
	cudaArray_t _volumeArrayInterleaved = nullptr;
	cudaTextureObject_t _volumeTextureInterleaved = 0;

    void allocGpu(ImageXYZC* img);
	void allocGpuInterleaved(ImageXYZC* img);
	void deallocGpu();

    void updateLutGpu(int channel, ImageXYZC* img);
};