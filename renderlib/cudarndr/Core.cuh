#pragma once

class CBoundingBox;
class CCamera;
class CDenoiseParams;
class CTiming;
class CVariance;
struct CudaCamera;
struct CudaLighting;
struct CRenderSettings;

struct cudaFB {
	float* fb;
	float* fbaccum;
	unsigned int* randomSeeds1;
	unsigned int* randomSeeds2;
};
#define MAX_CUDA_CHANNELS 4
struct cudaVolume {
	int nChannels;
	float intensityMax[MAX_CUDA_CHANNELS];
	float diffuse[MAX_CUDA_CHANNELS * 3];
	float specular[MAX_CUDA_CHANNELS * 3];
	float emissive[MAX_CUDA_CHANNELS * 3];
	float roughness[MAX_CUDA_CHANNELS];

	cudaTextureObject_t volumeTexture[MAX_CUDA_CHANNELS];
	cudaTextureObject_t gradientVolumeTexture[MAX_CUDA_CHANNELS];
	cudaTextureObject_t lutTexture[MAX_CUDA_CHANNELS];
	

	cudaVolume(int n) {
		nChannels = n;
		//volumeTexture = new cudaTextureObject_t[n];
		//gradientVolumeTexture = new cudaTextureObject_t[n];
		//lutTexture = new cudaTextureObject_t[n];
		//intensityMax = new float[n];
	}
	~cudaVolume() {
		//delete[] intensityMax;
		//delete[] volumeTexture;
		//delete[] gradientVolumeTexture;
		//delete[] lutTexture;
	}
};

void BindConstants(const CudaLighting& cudalt, const CDenoiseParams& denoise, const CudaCamera& cudacam, 
	const CBoundingBox& bbox, const CBoundingBox& clipped_bbox, const CRenderSettings& renderSettings, int numIterations,
	int w, int h, float gamma, float exposure);

void ComputeFocusDistance(const cudaVolume& volumedata);
void Render(const int& Type, int numExposures, int w, int h,
	cudaFB& framebuffers,
	const cudaVolume& volumedata,
	CTiming& RenderImage, CTiming& BlurImage, CTiming& PostProcessImage, CTiming& DenoiseImage, int& numIterations);
void ToneMap(float* inbuf, cudaSurfaceObject_t surfaceObj, int w, int h);
void Denoise(float* inbuf, cudaSurfaceObject_t surfaceObj, int w, int h, float lerpC);
