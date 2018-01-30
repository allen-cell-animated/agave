#pragma once

#include "Geometry.h"
#include "Timing.h"
#include "Scene.h"

class CCamera;
//class CScene;
class CVariance;
struct CudaLighting;

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

void BindConstants(CScene* pScene, const CudaLighting& cudalt, const CDenoiseParams& denoise, const CCamera& camera, const CBoundingBox& bbox);

// scene needs to be mutable to get nearest intersection for focusdist.
void Render(const int& Type, CCamera& camera,
	cudaFB& framebuffers,
	const cudaVolume& volumedata,
	CTiming& RenderImage, CTiming& BlurImage, CTiming& PostProcessImage, CTiming& DenoiseImage, int& numIterations);
void ToneMap(float* inbuf, cudaSurfaceObject_t surfaceObj, int w, int h);
void Denoise(float* inbuf, cudaSurfaceObject_t surfaceObj, int w, int h, float lerpC);
