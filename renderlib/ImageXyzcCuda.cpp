#include "ImageXyzcCuda.h"

#include "CudaUtilities.h"
#include "ImageXYZC.h"
#include "Logging.h"

#include <QElapsedTimer>
#include <QtDebug>

void ChannelCuda::allocGpu(ImageXYZC* img, int channel, bool do_volume, bool do_gradient_volume) {
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
		HandleCudaError(cudaMalloc3DArray(&_volumeArray, &channelDesc, volumeSize));

		_gpuBytes += (channelDesc.x + channelDesc.y + channelDesc.z + channelDesc.w)/8 * volumeSize.width * volumeSize.height * volumeSize.depth;

		// copy data to 3D array
		cudaMemcpy3DParms copyParams = { 0 };
		copyParams.srcPtr = make_cudaPitchedPtr(ch->_ptr, volumeSize.width*img->sizeOfElement(), volumeSize.width, volumeSize.height);
		copyParams.dstArray = _volumeArray;
		copyParams.extent = volumeSize;
		copyParams.kind = cudaMemcpyHostToDevice;
		HandleCudaError(cudaMemcpy3D(&copyParams));

		// create a texture object
		cudaResourceDesc texRes;
		memset(&texRes, 0, sizeof(cudaResourceDesc));
		texRes.resType = cudaResourceTypeArray;
		texRes.res.array.array = _volumeArray;
		cudaTextureDesc     texDescr;
		memset(&texDescr, 0, sizeof(cudaTextureDesc));
		texDescr.normalizedCoords = 1;
		texDescr.filterMode = cudaFilterModeLinear;
		texDescr.addressMode[0] = cudaAddressModeClamp;   // clamp
		texDescr.addressMode[1] = cudaAddressModeClamp;
		texDescr.addressMode[2] = cudaAddressModeClamp;
		texDescr.readMode = cudaReadModeNormalizedFloat;
		HandleCudaError(cudaCreateTextureObject(&_volumeTexture, &texRes, &texDescr, NULL));
	}

    // gradient buffer

	if (do_gradient_volume) {
		// assuming 16-bit data!
		cudaChannelFormatDesc gradientChannelDesc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned);

		// create 3D array
		HandleCudaError(cudaMalloc3DArray(&_volumeGradientArray, &gradientChannelDesc, volumeSize));
		_gpuBytes += (gradientChannelDesc.x + gradientChannelDesc.y + gradientChannelDesc.z + gradientChannelDesc.w) / 8 * volumeSize.width * volumeSize.height * volumeSize.depth;

		// copy data to 3D array
		cudaMemcpy3DParms gradientCopyParams = { 0 };
		gradientCopyParams.srcPtr = make_cudaPitchedPtr(ch->_gradientMagnitudePtr, volumeSize.width*img->sizeOfElement(), volumeSize.width, volumeSize.height);
		gradientCopyParams.dstArray = _volumeGradientArray;
		gradientCopyParams.extent = volumeSize;
		gradientCopyParams.kind = cudaMemcpyHostToDevice;
		HandleCudaError(cudaMemcpy3D(&gradientCopyParams));

		// create a texture object
		cudaResourceDesc gradientTexRes;
		memset(&gradientTexRes, 0, sizeof(cudaResourceDesc));
		gradientTexRes.resType = cudaResourceTypeArray;
		gradientTexRes.res.array.array = _volumeGradientArray;
		cudaTextureDesc     gradientTexDescr;
		memset(&gradientTexDescr, 0, sizeof(cudaTextureDesc));
		gradientTexDescr.normalizedCoords = 1;
		gradientTexDescr.filterMode = cudaFilterModeLinear;
		gradientTexDescr.addressMode[0] = cudaAddressModeClamp;   // clamp
		gradientTexDescr.addressMode[1] = cudaAddressModeClamp;
		gradientTexDescr.addressMode[2] = cudaAddressModeClamp;
		gradientTexDescr.readMode = cudaReadModeNormalizedFloat;
		HandleCudaError(cudaCreateTextureObject(&_volumeGradientTexture, &gradientTexRes, &gradientTexDescr, NULL));
	}

	// LUT buffer

	const int LUT_SIZE = 256;
	cudaChannelFormatDesc lutChannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	// create 1D array
	HandleCudaError(cudaMallocArray(&_volumeLutArray, &lutChannelDesc, LUT_SIZE, 1));
	_gpuBytes += (lutChannelDesc.x + lutChannelDesc.y + lutChannelDesc.z + lutChannelDesc.w)/8 * LUT_SIZE;
	// copy data to 1D array
	HandleCudaError(cudaMemcpyToArray(_volumeLutArray, 0, 0, ch->_lut, LUT_SIZE * 4, cudaMemcpyHostToDevice));

    // create texture objects
    cudaResourceDesc lutTexRes;
    memset(&lutTexRes, 0, sizeof(cudaResourceDesc));
    lutTexRes.resType = cudaResourceTypeArray;
    lutTexRes.res.array.array = _volumeLutArray;
    cudaTextureDesc     lutTexDescr;
    memset(&lutTexDescr, 0, sizeof(cudaTextureDesc));
    lutTexDescr.normalizedCoords = 1;
    lutTexDescr.filterMode = cudaFilterModeLinear;
    lutTexDescr.addressMode[0] = cudaAddressModeClamp;   // clamp
    lutTexDescr.addressMode[1] = cudaAddressModeClamp;
    lutTexDescr.addressMode[2] = cudaAddressModeClamp;
    lutTexDescr.readMode = cudaReadModeElementType;  // direct read the (filtered) value
    HandleCudaError(cudaCreateTextureObject(&_volumeLutTexture, &lutTexRes, &lutTexDescr, NULL));
}

void ChannelCuda::deallocGpu() {
    HandleCudaError(cudaDestroyTextureObject(_volumeLutTexture));
    _volumeLutTexture = 0;
    HandleCudaError(cudaDestroyTextureObject(_volumeGradientTexture));
    _volumeGradientTexture = 0;
    HandleCudaError(cudaDestroyTextureObject(_volumeTexture));
    _volumeTexture = 0;

    HandleCudaError(cudaFreeArray(_volumeLutArray));
    _volumeLutArray = nullptr;
    HandleCudaError(cudaFreeArray(_volumeGradientArray));
    _volumeGradientArray = nullptr;
    HandleCudaError(cudaFreeArray(_volumeArray));
    _volumeArray = nullptr;

	_gpuBytes = 0;
}

void ChannelCuda::updateLutGpu(int channel, ImageXYZC* img) {
    static const int LUT_SIZE = 256;
	HandleCudaError(cudaMemcpyToArray(_volumeLutArray, 0, 0, img->channel(channel)->_lut, LUT_SIZE * 4, cudaMemcpyHostToDevice));
}

void ImageCuda::allocGpu(ImageXYZC* img) {
    deallocGpu();
    _channels.clear(); 

	QElapsedTimer timer;
	timer.start();

    for (uint32_t i = 0; i < img->sizeC(); ++i) {
        ChannelCuda c;
		c._index = i;
		c.allocGpu(img, i);
        _channels.push_back(c);
    }

	LOG_DEBUG << "allocGPU: Image to GPU in " << timer.elapsed() << "ms";
}

void ImageCuda::createVolumeTexture4x16(ImageXYZC* img, cudaArray_t* deviceArray, cudaTextureObject_t* deviceTexture) {
	// assuming 16-bit data!
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindUnsigned);

	// create 3D array
	cudaExtent volumeSize;
	volumeSize.width = img->sizeX();
	volumeSize.height = img->sizeY();
	volumeSize.depth = img->sizeZ();
	HandleCudaError(cudaMalloc3DArray(deviceArray, &channelDesc, volumeSize));
	_gpuBytes += (channelDesc.x + channelDesc.y + channelDesc.z + channelDesc.w)/8 * volumeSize.width * volumeSize.height * volumeSize.depth;

	// create texture tied to array
	cudaResourceDesc texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = *deviceArray;
	cudaTextureDesc     texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords = 1;
	texDescr.filterMode = cudaFilterModeLinear;
	texDescr.addressMode[0] = cudaAddressModeClamp;   // clamp
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;
	texDescr.readMode = cudaReadModeNormalizedFloat;
	HandleCudaError(cudaCreateTextureObject(deviceTexture, &texRes, &texDescr, NULL));
}

void ImageCuda::updateVolumeData4x16(ImageXYZC* img, int c0, int c1, int c2, int c3)
{
	QElapsedTimer timer;
	timer.start();

	const int N = 4;
	int ch[4] = { c0, c1, c2, c3 };
	// interleaved all channels.
	// first 4.
	size_t xyz = img->sizeX()*img->sizeY()*img->sizeZ();
	uint16_t* v = new uint16_t[xyz * N];

	for (uint32_t i = 0; i < xyz; ++i) {
		for (int j = 0; j < N; ++j) {
			v[N * (i)+j] = img->channel(ch[j])->_ptr[(i)];
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
	copyParams.srcPtr = make_cudaPitchedPtr(v, volumeSize.width*img->sizeOfElement()*N, volumeSize.width, volumeSize.height);
	copyParams.dstArray = _volumeArrayInterleaved;
	copyParams.extent = volumeSize;
	copyParams.kind = cudaMemcpyHostToDevice;
	HandleCudaError(cudaMemcpy3D(&copyParams));

	LOG_DEBUG << "Copy volume to gpu: " << timer.elapsed() << "ms";


	delete[] v;
}

void ImageCuda::allocGpuInterleaved(ImageXYZC* img) {
	deallocGpu();
	_channels.clear();

	QElapsedTimer timer;
	timer.start();

	createVolumeTexture4x16(img, &_volumeArrayInterleaved, &_volumeTextureInterleaved);
	uint32_t numChannels = img->sizeC();
	updateVolumeData4x16(img, 0, std::min(1u, numChannels-1) , std::min(2u, numChannels - 1), std::min(3u, numChannels - 1));


	for (uint32_t i = 0; i < numChannels; ++i) {
		ChannelCuda c;
		c._index = i;
		c.allocGpu(img, i, false, false);
		_channels.push_back(c);

		_gpuBytes += c._gpuBytes;
	}

	LOG_DEBUG << "allocGPUinterleaved: Image to GPU in " << timer.elapsed() << "ms";
	LOG_DEBUG << "allocGPUinterleaved: GPU bytes: " << _gpuBytes;

}

void ImageCuda::deallocGpu() {
    for (size_t i = 0; i < _channels.size(); ++i) {
        _channels[i].deallocGpu();
    }
	HandleCudaError(cudaDestroyTextureObject(_volumeTextureInterleaved));
	_volumeTextureInterleaved = 0;
	HandleCudaError(cudaFreeArray(_volumeArrayInterleaved));
	_volumeArrayInterleaved = nullptr;

	_gpuBytes = 0;
}

void ImageCuda::updateLutGpu(int channel, ImageXYZC* img) {
    _channels[channel].updateLutGpu(channel, img);
}
