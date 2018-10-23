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
	size_t _gpuBytes = 0;

    void allocGpu(ImageXYZC* img, int channel, bool do_volume = true, bool do_gradient_volume = true);
    void deallocGpu();
	void updateLutGpu(int channel, ImageXYZC* img);

};

struct ImageCuda {
    std::vector<ChannelCuda> _channels;
	cudaArray_t _volumeArrayInterleaved = nullptr;
	cudaTextureObject_t _volumeTextureInterleaved = 0;

	size_t _gpuBytes = 0;
	
	// no one is calling this right now.
	// puts each channel into its own gpu volume buffer
	void allocGpu(ImageXYZC* img);

	// put first 4 channels into gpu array
	void allocGpuInterleaved(ImageXYZC* img);

	void deallocGpu();

    void updateLutGpu(int channel, ImageXYZC* img);

	void createVolumeTexture4x16(ImageXYZC* img, cudaArray_t* deviceArray, cudaTextureObject_t* deviceTexture);

	// similar to allocGpuInterleaved, change which channels are in the gpu volume buffer.
	void updateVolumeData4x16(ImageXYZC* img, int c0, int c1, int c2, int c3);
};