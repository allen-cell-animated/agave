#include "Logging.h"

#include "Core.cuh"
#include "Scene.h"
#include "helper_math.cuh"
#include "Camera2.cuh"
#include "Lighting2.cuh"
#include "Lighting2Impl.cuh"

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

//#include "Model.cuh"
#include "View.cuh"
#include "Blur.cuh"
#include "Denoise.cuh"
#include "Estimate.cuh"
#include "Utilities.cuh"
#include "SingleScattering.cuh"
#include "NearestIntersection.cuh"
//#include "SpecularBloom.cuh"
#include "ToneMap.cuh"

void Vec3ToFloat3(Vec3f* src, float3* dest) {
	dest->x = src->x;
	dest->y = src->y;
	dest->z = src->z;
}
void RGBToFloat3(CColorRgbHdr* src, float3* dest) {
	dest->x = src->r;
	dest->y = src->g;
	dest->z = src->b;
}

void FillCudaCamera(CCamera* pCamera, CudaCamera& c) {
	Vec3ToFloat3(&pCamera->m_From, &c.m_From);
	Vec3ToFloat3(&pCamera->m_N, &c.m_N);
	Vec3ToFloat3(&pCamera->m_U, &c.m_U);
	Vec3ToFloat3(&pCamera->m_V, &c.m_V);
	c.m_ApertureSize = pCamera->m_Aperture.m_Size;
	c.m_FocalDistance = pCamera->m_Focus.m_FocalDistance;
	c.m_InvScreen[0] = pCamera->m_Film.m_InvScreen.x;
	c.m_InvScreen[1] = pCamera->m_Film.m_InvScreen.y;
	c.m_Screen[0][0] = pCamera->m_Film.m_Screen[0][0];
	c.m_Screen[1][0] = pCamera->m_Film.m_Screen[1][0];
	c.m_Screen[0][1] = pCamera->m_Film.m_Screen[0][1];
	c.m_Screen[1][1] = pCamera->m_Film.m_Screen[1][1];
}

void BindConstants(CScene* pScene, const CudaLighting& cudalt)
{
	const float3 AaBbMin = make_float3(pScene->m_BoundingBox.GetMinP().x, pScene->m_BoundingBox.GetMinP().y, pScene->m_BoundingBox.GetMinP().z);
	const float3 AaBbMax = make_float3(pScene->m_BoundingBox.GetMaxP().x, pScene->m_BoundingBox.GetMaxP().y, pScene->m_BoundingBox.GetMaxP().z);

	HandleCudaError(cudaMemcpyToSymbol(gAaBbMin, &AaBbMin, sizeof(float3)));
	HandleCudaError(cudaMemcpyToSymbol(gAaBbMax, &AaBbMax, sizeof(float3)));

	const float3 InvAaBbMin = make_float3(pScene->m_BoundingBox.GetInvMinP().x, pScene->m_BoundingBox.GetInvMinP().y, pScene->m_BoundingBox.GetInvMinP().z);
	const float3 InvAaBbMax = make_float3(pScene->m_BoundingBox.GetInvMaxP().x, pScene->m_BoundingBox.GetInvMaxP().y, pScene->m_BoundingBox.GetInvMaxP().z);

	HandleCudaError(cudaMemcpyToSymbol(gInvAaBbMin, &InvAaBbMin, sizeof(float3)));
	HandleCudaError(cudaMemcpyToSymbol(gInvAaBbMax, &InvAaBbMax, sizeof(float3)));

	HandleCudaError(cudaMemcpyToSymbol(gShadingType, &pScene->m_ShadingType, sizeof(int)));
	HandleCudaError(cudaMemcpyToSymbol(gGradientFactor, &pScene->m_GradientFactor, sizeof(float)));

	const float StepSize		= pScene->m_StepSizeFactor * pScene->m_GradientDelta;
	const float StepSizeShadow	= pScene->m_StepSizeFactorShadow * pScene->m_GradientDelta;

	HandleCudaError(cudaMemcpyToSymbol(gStepSize, &StepSize, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gStepSizeShadow, &StepSizeShadow, sizeof(float)));

	const float DensityScale = pScene->m_DensityScale;

	HandleCudaError(cudaMemcpyToSymbol(gDensityScale, &DensityScale, sizeof(float)));
	
	const float GradientDelta		= 1.0f * pScene->m_GradientDelta;
	const float InvGradientDelta	= 1.0f / GradientDelta;
	const Vec3f GradientDeltaX(GradientDelta, 0.0f, 0.0f);
	const Vec3f GradientDeltaY(0.0f, GradientDelta, 0.0f);
	const Vec3f GradientDeltaZ(0.0f, 0.0f, GradientDelta);
	
	HandleCudaError(cudaMemcpyToSymbol(gGradientDelta, &GradientDelta, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gInvGradientDelta, &InvGradientDelta, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gGradientDeltaX, &GradientDeltaX, sizeof(float3)));
	HandleCudaError(cudaMemcpyToSymbol(gGradientDeltaY, &GradientDeltaY, sizeof(float3)));
	HandleCudaError(cudaMemcpyToSymbol(gGradientDeltaZ, &GradientDeltaZ, sizeof(float3)));
	
	const int FilmWidth		= pScene->m_Camera.m_Film.GetWidth();
	const int Filmheight	= pScene->m_Camera.m_Film.GetHeight();
	const int FilmNoPixels	= pScene->m_Camera.m_Film.m_Resolution.GetNoElements();

	HandleCudaError(cudaMemcpyToSymbol(gFilmWidth, &FilmWidth, sizeof(int)));
	HandleCudaError(cudaMemcpyToSymbol(gFilmHeight, &Filmheight, sizeof(int)));
	HandleCudaError(cudaMemcpyToSymbol(gFilmNoPixels, &FilmNoPixels, sizeof(int)));

	const int FilterWidth = 1;

	HandleCudaError(cudaMemcpyToSymbol(gFilterWidth, &FilterWidth, sizeof(int)));

	const float FilterWeights[10] = { 0.11411459588254977f, 0.08176668094332218f, 0.03008028089187349f, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

	HandleCudaError(cudaMemcpyToSymbol(gFilterWeights, FilterWeights, 10 * sizeof(float)));

	const float Gamma		= pScene->m_Camera.m_Film.m_Gamma;
	const float InvGamma	= 1.0f / Gamma;
	const float Exposure	= pScene->m_Camera.m_Film.m_Exposure;
	const float InvExposure	= 1.0f / Exposure;

	HandleCudaError(cudaMemcpyToSymbol(gExposure, &Exposure, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gInvExposure, &InvExposure, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gGamma, &Gamma, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gInvGamma, &InvGamma, sizeof(float)));

	const float denoiseEnabled = pScene->m_DenoiseParams.m_Enabled ? 1.0f : 0.0f;
	HandleCudaError(cudaMemcpyToSymbol(gDenoiseEnabled, &denoiseEnabled, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gDenoiseWindowRadius, &pScene->m_DenoiseParams.m_WindowRadius, sizeof(int)));
	HandleCudaError(cudaMemcpyToSymbol(gDenoiseInvWindowArea, &pScene->m_DenoiseParams.m_InvWindowArea, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gDenoiseNoise, &pScene->m_DenoiseParams.m_Noise, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gDenoiseWeightThreshold, &pScene->m_DenoiseParams.m_WeightThreshold, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gDenoiseLerpThreshold, &pScene->m_DenoiseParams.m_LerpThreshold, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gDenoiseLerpC, &pScene->m_DenoiseParams.m_LerpC, sizeof(float)));

	const float NoIterations	= pScene->GetNoIterations();
	const float InvNoIterations = 1.0f / ((NoIterations > 1.0f) ? NoIterations : 1.0f);

	HandleCudaError(cudaMemcpyToSymbol(gNoIterations, &NoIterations, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gInvNoIterations, &InvNoIterations, sizeof(float)));

	CudaCamera c;
	FillCudaCamera(&pScene->m_Camera, c);
	HandleCudaError(cudaMemcpyToSymbol(gCamera, &c, sizeof(CudaCamera)));

	HandleCudaError(cudaMemcpyToSymbol(gLighting, &cudalt, sizeof(CudaLighting)));
}

void Render(const int& Type, CScene* scene, CCamera& camera,
	cudaFB& framebuffers,
	const cudaVolume& volumedata,
	CTiming& RenderImage, CTiming& BlurImage, CTiming& PostProcessImage, CTiming& DenoiseImage)
{
	// find nearest intersection to set camera focal distance automatically.
	// then re-upload that data.
	if (camera.m_Focus.m_Type == 0) {
		camera.m_Focus.m_FocalDistance = NearestIntersection(volumedata);
		// send m_FocalDistance back to gpu.
		CudaCamera c;
		FillCudaCamera(&camera, c);
		HandleCudaError(cudaMemcpyToSymbol(gCamera, &c, sizeof(CudaCamera)));
	}

	for (int i = 0; i < camera.m_Film.m_ExposureIterations; ++i) {
		CCudaTimer TmrRender;

		switch (Type)
		{
		case 0:
		{
			SingleScattering(camera.m_Film.m_Resolution.GetResX(), camera.m_Film.m_Resolution.GetResY(),
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
		Estimate(camera.m_Film.m_Resolution.GetResX(), camera.m_Film.m_Resolution.GetResY(), 
			framebuffers.fb, framebuffers.fbaccum);
		PostProcessImage.AddDuration(TmrPostProcess.ElapsedTime());

		scene->SetNoIterations(scene->GetNoIterations() + 1);

		const float NoIterations = scene->GetNoIterations();
		const float InvNoIterations = 1.0f / ((NoIterations > 1.0f) ? NoIterations : 1.0f);
		HandleCudaError(cudaMemcpyToSymbol(gNoIterations, &NoIterations, sizeof(float)));
		HandleCudaError(cudaMemcpyToSymbol(gInvNoIterations, &InvNoIterations, sizeof(float)));
	}
}
