#pragma once

#include "Geometry.h"
#include "Timing.h"
#include "Scene.h"

class CScene;
class CVariance;

struct cudaFB {
	float* fb;
	float* fbaccum;
	unsigned int* randomSeeds1;
	unsigned int* randomSeeds2;
};
#define MAX_CHANNELS 8
struct cudaVolume {
	int nChannels;
	float intensityMax[MAX_CHANNELS];
	float diffuse[MAX_CHANNELS * 3];

	cudaTextureObject_t volumeTexture[MAX_CHANNELS];
	cudaTextureObject_t gradientVolumeTexture[MAX_CHANNELS];
	cudaTextureObject_t lutTexture[MAX_CHANNELS];
	

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

void BindDensityBuffer(short* pBuffer, cudaExtent Extent);
void BindGradientMagnitudeBuffer(short* pBuffer, cudaExtent Extent);
void UnbindDensityBuffer(void);
void UnbindGradientMagnitudeBuffer(void);
void BindRenderCanvasView(const CResolution2D& Resolution);
void ResetRenderCanvasView(void);
void FreeRenderCanvasView(void);
unsigned char* GetDisplayEstimate(void);
//void BindTransferFunctionOpacity(CTransferFunction& TransferFunctionOpacity);
//void BindTransferFunctionDiffuse(CTransferFunction& TransferFunctionDiffuse);
//void BindTransferFunctionSpecular(CTransferFunction& TransferFunctionSpecular);
//void BindTransferFunctionRoughness(CTransferFunction& TransferFunctionRoughness);
//void BindTransferFunctionEmission(CTransferFunction& TransferFunctionEmission);
//void UnbindTransferFunctionOpacity(void);
//void UnbindTransferFunctionDiffuse(void);
//void UnbindTransferFunctionSpecular(void);
//void UnbindTransferFunctionRoughness(void);
//void UnbindTransferFunctionEmission(void);
void BindConstants(CScene* pScene);
//void Render(const int& Type, CScene& Scene, CTiming& RenderImage, CTiming& BlurImage, CTiming& PostProcessImage, CTiming& DenoiseImage);
// scene needs to be mutable to get nearest intersection for focusdist.
void Render(const int& Type, CScene& Scene,
	cudaFB& framebuffers,
	const cudaVolume& volumedata,
	CTiming& RenderImage, CTiming& BlurImage, CTiming& PostProcessImage, CTiming& DenoiseImage);
void ToneMap(float* inbuf, cudaSurfaceObject_t surfaceObj, int w, int h);
void Denoise(float* inbuf, cudaSurfaceObject_t surfaceObj, int w, int h);
