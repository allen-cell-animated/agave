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
struct cudaVolume {
	cudaTextureObject_t volumeTexture;
	cudaTextureObject_t gradientVolumeTexture;
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
