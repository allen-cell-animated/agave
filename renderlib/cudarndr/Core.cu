#include "Logging.h"

#include "Core.cuh"
//#include "Scene.h"
//#include "helper_math.cuh"
#include "Camera2.cuh"
#include "Camera2Impl.cuh"
#include "Lighting2.cuh"
#include "Lighting2Impl.cuh"
#include "DenoiseParams.cuh"

CD float3		gAaBbMin;
CD float3		gAaBbMax;
CD float3		gInvAaBbMin;
CD float3		gInvAaBbMax;
CD float		gStepSize;
CD float		gStepSizeShadow;
CD float		gDensityScale;
CD float		gGradientDelta;
CD float		gInvGradientDelta;
CD float3		gGradientDeltaX;
CD float3		gGradientDeltaY;
CD float3		gGradientDeltaZ;
CD int			gFilmWidth;
CD int			gFilmHeight;
CD int			gFilmNoPixels;
CD int			gFilterWidth;
CD float		gFilterWeights[10];
CD float		gExposure;
CD float		gInvExposure;
CD float		gGamma;
CD float		gInvGamma;
CD float		gDenoiseEnabled;
CD int		gDenoiseWindowRadius;
CD float		gDenoiseInvWindowArea;
CD float		gDenoiseNoise;
CD float		gDenoiseWeightThreshold;
CD float		gDenoiseLerpThreshold;
CD float		gDenoiseLerpC;
CD float		gNoIterations;
CD float		gInvNoIterations;

CD int gShadingType;
CD float gGradientFactor;

CD CudaLighting gLighting;

// enough data to generate a camera ray
CD CudaCamera gCamera;

#define TF_NO_SAMPLES		128
#define INV_TF_NO_SAMPLES	1.0f / (float)TF_NO_SAMPLES

//#include "Camera.cuh"
//#include "Model.cuh"
//#include "View.cuh"
//#include "Blur.cuh"
#include "Denoise.cuh"
#include "Estimate.cuh"
#include "Utilities.cuh"
#include "SingleScattering.cuh"
#include "NearestIntersection.cuh"
//#include "SpecularBloom.cuh"
#include "ToneMap.cuh"

void RGBToFloat3(const CColorRgbHdr* src, float3* dest) {
	dest->x = src->r;
	dest->y = src->g;
	dest->z = src->b;
}


void BindConstants(const CudaLighting& cudalt, const CDenoiseParams& denoise, const CudaCamera& cudacam, 
	const CBoundingBox& bbox, const CRenderSettings& renderSettings, int numIterations,
	int w, int h, float gamma, float exposure)
{
	const float3 AaBbMin = make_float3(bbox.GetMinP().x, bbox.GetMinP().y, bbox.GetMinP().z);
	const float3 AaBbMax = make_float3(bbox.GetMaxP().x, bbox.GetMaxP().y, bbox.GetMaxP().z);

	HandleCudaError(cudaMemcpyToSymbol(gAaBbMin, &AaBbMin, sizeof(float3)));
	HandleCudaError(cudaMemcpyToSymbol(gAaBbMax, &AaBbMax, sizeof(float3)));

	const float3 InvAaBbMin = make_float3(bbox.GetInvMinP().x, bbox.GetInvMinP().y, bbox.GetInvMinP().z);
	const float3 InvAaBbMax = make_float3(bbox.GetInvMaxP().x, bbox.GetInvMaxP().y, bbox.GetInvMaxP().z);

	HandleCudaError(cudaMemcpyToSymbol(gInvAaBbMin, &InvAaBbMin, sizeof(float3)));
	HandleCudaError(cudaMemcpyToSymbol(gInvAaBbMax, &InvAaBbMax, sizeof(float3)));

	HandleCudaError(cudaMemcpyToSymbol(gShadingType, &renderSettings.m_ShadingType, sizeof(int)));
	HandleCudaError(cudaMemcpyToSymbol(gGradientFactor, &renderSettings.m_GradientFactor, sizeof(float)));

	const float StepSize		= renderSettings.m_StepSizeFactor * renderSettings.m_GradientDelta;
	const float StepSizeShadow	= renderSettings.m_StepSizeFactorShadow * renderSettings.m_GradientDelta;

	HandleCudaError(cudaMemcpyToSymbol(gStepSize, &StepSize, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gStepSizeShadow, &StepSizeShadow, sizeof(float)));

	const float DensityScale = renderSettings.m_DensityScale;

	HandleCudaError(cudaMemcpyToSymbol(gDensityScale, &DensityScale, sizeof(float)));
	
	const float GradientDelta		= 1.0f * renderSettings.m_GradientDelta;
	const float InvGradientDelta	= 1.0f / GradientDelta;
	const Vec3f GradientDeltaX(GradientDelta, 0.0f, 0.0f);
	const Vec3f GradientDeltaY(0.0f, GradientDelta, 0.0f);
	const Vec3f GradientDeltaZ(0.0f, 0.0f, GradientDelta);
	
	HandleCudaError(cudaMemcpyToSymbol(gGradientDelta, &GradientDelta, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gInvGradientDelta, &InvGradientDelta, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gGradientDeltaX, &GradientDeltaX, sizeof(float3)));
	HandleCudaError(cudaMemcpyToSymbol(gGradientDeltaY, &GradientDeltaY, sizeof(float3)));
	HandleCudaError(cudaMemcpyToSymbol(gGradientDeltaZ, &GradientDeltaZ, sizeof(float3)));
	
	const int FilmWidth		= w;
	const int Filmheight	= h;
	//const int FilmNoPixels	= camera.m_Film.m_Resolution.GetNoElements();

	HandleCudaError(cudaMemcpyToSymbol(gFilmWidth, &FilmWidth, sizeof(int)));
	HandleCudaError(cudaMemcpyToSymbol(gFilmHeight, &Filmheight, sizeof(int)));
	//HandleCudaError(cudaMemcpyToSymbol(gFilmNoPixels, &FilmNoPixels, sizeof(int)));

	const int FilterWidth = 1;

	HandleCudaError(cudaMemcpyToSymbol(gFilterWidth, &FilterWidth, sizeof(int)));

	const float FilterWeights[10] = { 0.11411459588254977f, 0.08176668094332218f, 0.03008028089187349f, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

	HandleCudaError(cudaMemcpyToSymbol(gFilterWeights, FilterWeights, 10 * sizeof(float)));

	const float Gamma		= gamma;
	const float InvGamma	= 1.0f / Gamma;
	const float Exposure	= exposure;
	const float InvExposure	= 1.0f / Exposure;

	HandleCudaError(cudaMemcpyToSymbol(gExposure, &Exposure, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gInvExposure, &InvExposure, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gGamma, &Gamma, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gInvGamma, &InvGamma, sizeof(float)));

	const float denoiseEnabled = denoise.m_Enabled ? 1.0f : 0.0f;
	HandleCudaError(cudaMemcpyToSymbol(gDenoiseEnabled, &denoiseEnabled, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gDenoiseWindowRadius, &denoise.m_WindowRadius, sizeof(int)));
	HandleCudaError(cudaMemcpyToSymbol(gDenoiseInvWindowArea, &denoise.m_InvWindowArea, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gDenoiseNoise, &denoise.m_Noise, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gDenoiseWeightThreshold, &denoise.m_WeightThreshold, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gDenoiseLerpThreshold, &denoise.m_LerpThreshold, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gDenoiseLerpC, &denoise.m_LerpC, sizeof(float)));

	const float NoIterations	= numIterations;
	const float InvNoIterations = 1.0f / ((NoIterations > 1.0f) ? NoIterations : 1.0f);

	HandleCudaError(cudaMemcpyToSymbol(gNoIterations, &NoIterations, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gInvNoIterations, &InvNoIterations, sizeof(float)));

	HandleCudaError(cudaMemcpyToSymbol(gCamera, &cudacam, sizeof(CudaCamera)));

	HandleCudaError(cudaMemcpyToSymbol(gLighting, &cudalt, sizeof(CudaLighting)));
}

// BindConstants must be called first to initialize vars used by kernels
void Render(const int& Type, int numExposures, int w, int h,
	cudaFB& framebuffers,
	const cudaVolume& volumedata,
	CTiming& RenderImage, CTiming& BlurImage, CTiming& PostProcessImage, CTiming& DenoiseImage,
	int& numIterations)
{
	// find nearest intersection to set camera focal distance automatically.
	// then re-upload that data.
	//if (camera.m_Focus.m_Type == 0) {
		float fd = NearestIntersection(volumedata);
		//camera.m_Focus.m_FocalDistance = NearestIntersection(volumedata);
		// send m_FocalDistance back to gpu.
		//CudaCamera c;
		//FillCudaCamera(&camera, c);
		HandleCudaError(cudaMemcpyToSymbol(gCamera, &fd, sizeof(float)));
	//}

	for (int i = 0; i < numExposures; ++i) {
		CCudaTimer TmrRender;

		switch (Type)
		{
		case 0:
		{
			SingleScattering(w, h,
				volumedata, framebuffers.fb, framebuffers.randomSeeds1, framebuffers.randomSeeds2);
			break;
		}

		case 1:
		{
			//			MultipleScattering(&Scene);
			break;
		}
		}
		RenderImage.AddDuration(TmrRender.ElapsedTime());

		// estimate just adds to accumulation buffer.
		CCudaTimer TmrPostProcess;
		Estimate(w, h, 
			framebuffers.fb, framebuffers.fbaccum);
		PostProcessImage.AddDuration(TmrPostProcess.ElapsedTime());

		numIterations++;
		const float NoIterations = numIterations;
		const float InvNoIterations = 1.0f / ((NoIterations > 1.0f) ? NoIterations : 1.0f);
		HandleCudaError(cudaMemcpyToSymbol(gNoIterations, &NoIterations, sizeof(float)));
		HandleCudaError(cudaMemcpyToSymbol(gInvNoIterations, &InvNoIterations, sizeof(float)));
	}
}
