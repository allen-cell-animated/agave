#pragma once

#include "glad/include/glad/glad.h"

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
	
	void allocGpu(ImageXYZC* img);
	void allocGpuInterleaved(ImageXYZC* img);
	void deallocGpu();

    void updateLutGpu(int channel, ImageXYZC* img);

	void createVolumeTexture4x16(ImageXYZC* img, cudaArray_t* deviceArray, cudaTextureObject_t* deviceTexture);
	void updateVolumeData4x16(ImageXYZC* img, int c0, int c1, int c2, int c3);
};

struct ChannelGL {

    GLuint _volumeTexture = 0;
    GLuint _volumeGradientTexture = 0;
    GLuint _volumeLutTexture = 0;

    int _index;
    size_t _gpuBytes = 0;

    void allocGpu(ImageXYZC* img, int channel);
    void deallocGpu();
    void updateLutGpu(int channel, ImageXYZC* img);

};

struct ImageGL {
    std::vector<ChannelCuda> _channels;
    GLuint _volumeTextureInterleaved = 0;

    size_t _gpuBytes = 0;

    void allocGpuInterleaved(ImageXYZC* img);
    void deallocGpu();

    void updateLutGpu(int channel, ImageXYZC* img);

    void createVolumeTexture4x16(ImageXYZC* img, GLuint* deviceTexture);
    void updateVolumeData4x16(ImageXYZC* img, int c0, int c1, int c2, int c3);

};