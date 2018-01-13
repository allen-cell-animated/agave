#include "Logging.h"

#include "Core.cuh"
#include "helper_math.cuh"
#include "Camera2.cuh"
#include "Lighting2.cuh"

//texture<short, cudaTextureType3D, cudaReadModeNormalizedFloat>		gTexDensity;
//texture<short, cudaTextureType3D, cudaReadModeNormalizedFloat>		gTexGradientMagnitude;

cudaTextureObject_t gTexDensity;
cudaTextureObject_t gTexGradientMagnitude;

//texture<float, cudaTextureType3D, cudaReadModeElementType>			gTexExtinction;
//texture<float, cudaTextureType1D, cudaReadModeElementType>			gTexOpacity;
//texture<float4, cudaTextureType1D, cudaReadModeElementType>			gTexDiffuse;
//texture<float4, cudaTextureType1D, cudaReadModeElementType>			gTexSpecular;
//texture<float, cudaTextureType1D, cudaReadModeElementType>			gTexRoughness;
//texture<float4, cudaTextureType1D, cudaReadModeElementType>			gTexEmission;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat>		gTexRunningEstimateRgba;

cudaArray* gpDensityArray				= NULL;
cudaArray* gpGradientMagnitudeArray		= NULL;
//cudaArray* gpOpacityArray				= NULL;
//cudaArray* gpDiffuseArray				= NULL;
//cudaArray* gpSpecularArray				= NULL;
//cudaArray* gpRoughnessArray				= NULL;
//cudaArray* gpEmissionArray				= NULL;

CD float3		gAaBbMin;
CD float3		gAaBbMax;
CD float3		gInvAaBbMin;
CD float3		gInvAaBbMax;
CD float		gIntensityMin;
CD float		gIntensityMax;
CD float		gIntensityRange;
CD float		gIntensityInvRange;
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

CD float4		gDiffuseColor;
CD float4		gSpecularColor;
CD float4		gEmissiveColor;

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

CCudaView	gRenderCanvasView;

void BindDensityBuffer(short* pBuffer, cudaExtent volumeSize)
{
	// create 3D array
	cudaChannelFormatDesc gradientChannelDesc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned);
	HandleCudaError(cudaMalloc3DArray(&gpDensityArray, &gradientChannelDesc, volumeSize));

	// copy data to 3D array
	cudaMemcpy3DParms gradientCopyParams = { 0 };
	gradientCopyParams.srcPtr = make_cudaPitchedPtr(pBuffer, volumeSize.width*sizeof(short), volumeSize.width, volumeSize.height);
	gradientCopyParams.dstArray = gpDensityArray;
	gradientCopyParams.extent = volumeSize;
	gradientCopyParams.kind = cudaMemcpyHostToDevice;
	HandleCudaError(cudaMemcpy3D(&gradientCopyParams));

	cudaResourceDesc gradientTexRes;
	memset(&gradientTexRes, 0, sizeof(cudaResourceDesc));
	gradientTexRes.resType = cudaResourceTypeArray;
	gradientTexRes.res.array.array = gpDensityArray;
	cudaTextureDesc     gradientTexDescr;
	memset(&gradientTexDescr, 0, sizeof(cudaTextureDesc));
	gradientTexDescr.normalizedCoords = 1;
	gradientTexDescr.filterMode = cudaFilterModeLinear;
	gradientTexDescr.addressMode[0] = cudaAddressModeClamp;   // clamp
	gradientTexDescr.addressMode[1] = cudaAddressModeClamp;
	gradientTexDescr.addressMode[2] = cudaAddressModeClamp;
	gradientTexDescr.readMode = cudaReadModeNormalizedFloat;
	HandleCudaError(cudaCreateTextureObject(&gTexDensity, &gradientTexRes, &gradientTexDescr, NULL));
}

void BindGradientMagnitudeBuffer(short* pBuffer, cudaExtent volumeSize)
{
	// create 3D array
	cudaChannelFormatDesc gradientChannelDesc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned);
	HandleCudaError(cudaMalloc3DArray(&gpGradientMagnitudeArray, &gradientChannelDesc, volumeSize));

	// copy data to 3D array
	cudaMemcpy3DParms gradientCopyParams = { 0 };
	gradientCopyParams.srcPtr = make_cudaPitchedPtr(pBuffer, volumeSize.width*sizeof(short), volumeSize.width, volumeSize.height);
	gradientCopyParams.dstArray = gpGradientMagnitudeArray;
	gradientCopyParams.extent = volumeSize;
	gradientCopyParams.kind = cudaMemcpyHostToDevice;
	HandleCudaError(cudaMemcpy3D(&gradientCopyParams));

	cudaResourceDesc gradientTexRes;
	memset(&gradientTexRes, 0, sizeof(cudaResourceDesc));
	gradientTexRes.resType = cudaResourceTypeArray;
	gradientTexRes.res.array.array = gpGradientMagnitudeArray;
	cudaTextureDesc     gradientTexDescr;
	memset(&gradientTexDescr, 0, sizeof(cudaTextureDesc));
	gradientTexDescr.normalizedCoords = 1;
	gradientTexDescr.filterMode = cudaFilterModeLinear;
	gradientTexDescr.addressMode[0] = cudaAddressModeClamp;   // clamp
	gradientTexDescr.addressMode[1] = cudaAddressModeClamp;
	gradientTexDescr.addressMode[2] = cudaAddressModeClamp;
	gradientTexDescr.readMode = cudaReadModeNormalizedFloat;
	HandleCudaError(cudaCreateTextureObject(&gTexGradientMagnitude, &gradientTexRes, &gradientTexDescr, NULL));
}

void UnbindDensityBuffer(void)
{
	HandleCudaError(cudaFreeArray(gpDensityArray));
	gpDensityArray = NULL;
	HandleCudaError(cudaDestroyTextureObject(gTexDensity));
}

void UnbindGradientMagnitudeBuffer(void)
{
	HandleCudaError(cudaFreeArray(gpGradientMagnitudeArray));
	gpGradientMagnitudeArray = NULL;
	HandleCudaError(cudaDestroyTextureObject(gTexGradientMagnitude));
}

void BindRenderCanvasView(const CResolution2D& Resolution)
{
	gRenderCanvasView.Resize(Resolution);

	cudaChannelFormatDesc Channel;
	
	Channel = cudaCreateChannelDesc<uchar4>();

	HandleCudaError(cudaBindTexture2D(0, gTexRunningEstimateRgba, gRenderCanvasView.m_EstimateRgbaLdr.GetPtr(), Channel, gRenderCanvasView.GetWidth(), gRenderCanvasView.GetHeight(), gRenderCanvasView.m_EstimateRgbaLdr.GetPitch()));
}

void ResetRenderCanvasView(void)
{
	gRenderCanvasView.Reset();
}

void FreeRenderCanvasView(void)
{
	gRenderCanvasView.Free();
}

unsigned char* GetDisplayEstimate(void)
{
	return (unsigned char*)gRenderCanvasView.m_DisplayEstimateRgbLdr.GetPtr(0, 0);
}
#if 0
void BindTransferFunctionOpacity(CTransferFunction& TransferFunctionOpacity)
{
	gTexOpacity.normalized		= true;
	gTexOpacity.filterMode		= cudaFilterModeLinear;
	gTexOpacity.addressMode[0]	= cudaAddressModeClamp;

	float Opacity[TF_NO_SAMPLES];

	for (int i = 0; i < TF_NO_SAMPLES; i++)
		Opacity[i] = TransferFunctionOpacity.F((float)i * INV_TF_NO_SAMPLES).r;
	
	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<float>();

	if (gpOpacityArray == NULL)
		HandleCudaError(cudaMallocArray(&gpOpacityArray, &ChannelDesc, TF_NO_SAMPLES, 1));

	HandleCudaError(cudaMemcpyToArray(gpOpacityArray, 0, 0, Opacity, TF_NO_SAMPLES * sizeof(float), cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexOpacity, gpOpacityArray, ChannelDesc));
}

void UnbindTransferFunctionOpacity(void)
{
	HandleCudaError(cudaFreeArray(gpOpacityArray));
	gpOpacityArray = NULL;
	HandleCudaError(cudaUnbindTexture(gTexOpacity));
}

void BindTransferFunctionDiffuse(CTransferFunction& TransferFunctionDiffuse)
{
	gTexDiffuse.normalized		= true;
	gTexDiffuse.filterMode		= cudaFilterModeLinear;
	gTexDiffuse.addressMode[0]	= cudaAddressModeClamp;

	float4 Diffuse[TF_NO_SAMPLES];

	for (int i = 0; i < TF_NO_SAMPLES; i++)
	{
		Diffuse[i].x = TransferFunctionDiffuse.F((float)i * INV_TF_NO_SAMPLES).r;
		Diffuse[i].y = TransferFunctionDiffuse.F((float)i * INV_TF_NO_SAMPLES).g;
		Diffuse[i].z = TransferFunctionDiffuse.F((float)i * INV_TF_NO_SAMPLES).b;
	}

	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<float4>();
	
	if (gpDiffuseArray == NULL)
		HandleCudaError(cudaMallocArray(&gpDiffuseArray, &ChannelDesc, TF_NO_SAMPLES, 1));

	HandleCudaError(cudaMemcpyToArray(gpDiffuseArray, 0, 0, Diffuse, TF_NO_SAMPLES * sizeof(float4), cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexDiffuse, gpDiffuseArray, ChannelDesc));
}

void UnbindTransferFunctionDiffuse(void)
{
	HandleCudaError(cudaFreeArray(gpDiffuseArray));
	gpDiffuseArray = NULL;
	HandleCudaError(cudaUnbindTexture(gTexDiffuse));
}

void BindTransferFunctionSpecular(CTransferFunction& TransferFunctionSpecular)
{
	gTexSpecular.normalized		= true;
	gTexSpecular.filterMode		= cudaFilterModeLinear;
	gTexSpecular.addressMode[0]	= cudaAddressModeClamp;

	float4 Specular[TF_NO_SAMPLES];

	for (int i = 0; i < TF_NO_SAMPLES; i++)
	{
		Specular[i].x = TransferFunctionSpecular.F((float)i * INV_TF_NO_SAMPLES).r;
		Specular[i].y = TransferFunctionSpecular.F((float)i * INV_TF_NO_SAMPLES).g;
		Specular[i].z = TransferFunctionSpecular.F((float)i * INV_TF_NO_SAMPLES).b;
	}

	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<float4>();
	
	if (gpSpecularArray == NULL)
		HandleCudaError(cudaMallocArray(&gpSpecularArray, &ChannelDesc, TF_NO_SAMPLES, 1));

	HandleCudaError(cudaMemcpyToArray(gpSpecularArray, 0, 0, Specular, TF_NO_SAMPLES * sizeof(float4), cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexSpecular, gpSpecularArray, ChannelDesc));
}

void UnbindTransferFunctionSpecular(void)
{
	HandleCudaError(cudaFreeArray(gpSpecularArray));
	gpSpecularArray = NULL;
	HandleCudaError(cudaUnbindTexture(gTexSpecular));
}

void BindTransferFunctionRoughness(CTransferFunction& TransferFunctionRoughness)
{
	gTexRoughness.normalized		= true;
	gTexRoughness.filterMode		= cudaFilterModeLinear;
	gTexRoughness.addressMode[0]	= cudaAddressModeClamp;

	float Roughness[TF_NO_SAMPLES];

	for (int i = 0; i < TF_NO_SAMPLES; i++)
		Roughness[i] = TransferFunctionRoughness.F((float)i * INV_TF_NO_SAMPLES).r;
	
	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<float>();

	if (gpRoughnessArray == NULL)
		HandleCudaError(cudaMallocArray(&gpRoughnessArray, &ChannelDesc, TF_NO_SAMPLES, 1));

	HandleCudaError(cudaMemcpyToArray(gpRoughnessArray, 0, 0, Roughness, TF_NO_SAMPLES * sizeof(float),  cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexRoughness, gpRoughnessArray, ChannelDesc));
}

void UnbindTransferFunctionRoughness(void)
{
	HandleCudaError(cudaFreeArray(gpRoughnessArray));
	gpRoughnessArray = NULL;
	HandleCudaError(cudaUnbindTexture(gTexRoughness));
}

void BindTransferFunctionEmission(CTransferFunction& TransferFunctionEmission)
{
	gTexEmission.normalized		= true;
	gTexEmission.filterMode		= cudaFilterModeLinear;
	gTexEmission.addressMode[0]	= cudaAddressModeClamp;

	float4 Emission[TF_NO_SAMPLES];

	for (int i = 0; i < TF_NO_SAMPLES; i++)
	{
		Emission[i].x = TransferFunctionEmission.F((float)i * INV_TF_NO_SAMPLES).r;
		Emission[i].y = TransferFunctionEmission.F((float)i * INV_TF_NO_SAMPLES).g;
		Emission[i].z = TransferFunctionEmission.F((float)i * INV_TF_NO_SAMPLES).b;
	}

	cudaChannelFormatDesc ChannelDesc = cudaCreateChannelDesc<float4>();
	
	if (gpEmissionArray == NULL)
		HandleCudaError(cudaMallocArray(&gpEmissionArray, &ChannelDesc, TF_NO_SAMPLES, 1));

	HandleCudaError(cudaMemcpyToArray(gpEmissionArray, 0, 0, Emission, TF_NO_SAMPLES * sizeof(float4),  cudaMemcpyHostToDevice));
	HandleCudaError(cudaBindTextureToArray(gTexEmission, gpEmissionArray, ChannelDesc));
}

void UnbindTransferFunctionEmission(void)
{
	HandleCudaError(cudaFreeArray(gpEmissionArray));
	gpEmissionArray = NULL;
	HandleCudaError(cudaUnbindTexture(gTexEmission));
}
#endif

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

void FillCudaCamera(CScene* pScene, CudaCamera& c) {
	Vec3ToFloat3(&pScene->m_Camera.m_From, &c.m_From);
	Vec3ToFloat3(&pScene->m_Camera.m_N, &c.m_N);
	Vec3ToFloat3(&pScene->m_Camera.m_U, &c.m_U);
	Vec3ToFloat3(&pScene->m_Camera.m_V, &c.m_V);
	c.m_ApertureSize = pScene->m_Camera.m_Aperture.m_Size;
	c.m_FocalDistance = pScene->m_Camera.m_Focus.m_FocalDistance;
	c.m_InvScreen[0] = pScene->m_Camera.m_Film.m_InvScreen.x;
	c.m_InvScreen[1] = pScene->m_Camera.m_Film.m_InvScreen.y;
	c.m_Screen[0][0] = pScene->m_Camera.m_Film.m_Screen[0][0];
	c.m_Screen[1][0] = pScene->m_Camera.m_Film.m_Screen[1][0];
	c.m_Screen[0][1] = pScene->m_Camera.m_Film.m_Screen[0][1];
	c.m_Screen[1][1] = pScene->m_Camera.m_Film.m_Screen[1][1];
}

void BindConstants(CScene* pScene, CScene* pDeviceScene)
{
	const float3 AaBbMin = make_float3(pScene->m_BoundingBox.GetMinP().x, pScene->m_BoundingBox.GetMinP().y, pScene->m_BoundingBox.GetMinP().z);
	const float3 AaBbMax = make_float3(pScene->m_BoundingBox.GetMaxP().x, pScene->m_BoundingBox.GetMaxP().y, pScene->m_BoundingBox.GetMaxP().z);

	HandleCudaError(cudaMemcpyToSymbol(gAaBbMin, &AaBbMin, sizeof(float3)));
	HandleCudaError(cudaMemcpyToSymbol(gAaBbMax, &AaBbMax, sizeof(float3)));

	const float3 InvAaBbMin = make_float3(pScene->m_BoundingBox.GetInvMinP().x, pScene->m_BoundingBox.GetInvMinP().y, pScene->m_BoundingBox.GetInvMinP().z);
	const float3 InvAaBbMax = make_float3(pScene->m_BoundingBox.GetInvMaxP().x, pScene->m_BoundingBox.GetInvMaxP().y, pScene->m_BoundingBox.GetInvMaxP().z);

	HandleCudaError(cudaMemcpyToSymbol(gInvAaBbMin, &InvAaBbMin, sizeof(float3)));
	HandleCudaError(cudaMemcpyToSymbol(gInvAaBbMax, &InvAaBbMax, sizeof(float3)));

	HandleCudaError(cudaMemcpyToSymbol(gDiffuseColor, pScene->m_DiffuseColor, sizeof(float4)));
	HandleCudaError(cudaMemcpyToSymbol(gSpecularColor, pScene->m_SpecularColor, sizeof(float4)));
	HandleCudaError(cudaMemcpyToSymbol(gEmissiveColor, pScene->m_EmissiveColor, sizeof(float4)));

	HandleCudaError(cudaMemcpyToSymbol(gShadingType, &pScene->m_ShadingType, sizeof(int)));
	HandleCudaError(cudaMemcpyToSymbol(gGradientFactor, &pScene->m_GradientFactor, sizeof(float)));

	const float IntensityMin		= pScene->m_IntensityRange.GetMin();
	const float IntensityMax		= pScene->m_IntensityRange.GetMax();
	const float IntensityRange		= pScene->m_IntensityRange.GetRange();
	const float IntensityInvRange	= 1.0f / IntensityRange;

	HandleCudaError(cudaMemcpyToSymbol(gIntensityMin, &IntensityMin, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gIntensityMax, &IntensityMax, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gIntensityRange, &IntensityRange, sizeof(float)));
	HandleCudaError(cudaMemcpyToSymbol(gIntensityInvRange, &IntensityInvRange, sizeof(float)));

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
	FillCudaCamera(pScene, c);
	HandleCudaError(cudaMemcpyToSymbol(gCamera, &c, sizeof(CudaCamera)));
	
	CudaLighting cl;
	cl.m_NoLights = pScene->m_Lighting.m_NoLights;
	for (int i = 0; i < cl.m_NoLights; ++i) {
		cl.m_Lights[i].m_Theta = pScene->m_Lighting.m_Lights[i].m_Theta;
		cl.m_Lights[i].m_Phi = pScene->m_Lighting.m_Lights[i].m_Phi;
		cl.m_Lights[i].m_Width = pScene->m_Lighting.m_Lights[i].m_Width;
		cl.m_Lights[i].m_InvWidth = pScene->m_Lighting.m_Lights[i].m_InvWidth;
		cl.m_Lights[i].m_HalfWidth = pScene->m_Lighting.m_Lights[i].m_HalfWidth;
		cl.m_Lights[i].m_InvHalfWidth = pScene->m_Lighting.m_Lights[i].m_InvHalfWidth;
		cl.m_Lights[i].m_Height = pScene->m_Lighting.m_Lights[i].m_Height;
		cl.m_Lights[i].m_InvHeight = pScene->m_Lighting.m_Lights[i].m_InvHeight;
		cl.m_Lights[i].m_HalfHeight = pScene->m_Lighting.m_Lights[i].m_HalfHeight;
		cl.m_Lights[i].m_InvHalfHeight = pScene->m_Lighting.m_Lights[i].m_InvHalfHeight;
		cl.m_Lights[i].m_Distance = pScene->m_Lighting.m_Lights[i].m_Distance;
		cl.m_Lights[i].m_SkyRadius = pScene->m_Lighting.m_Lights[i].m_SkyRadius;
		Vec3ToFloat3(&pScene->m_Lighting.m_Lights[i].m_P, &cl.m_Lights[i].m_P);
		Vec3ToFloat3(&pScene->m_Lighting.m_Lights[i].m_Target, &cl.m_Lights[i].m_Target);
		Vec3ToFloat3(&pScene->m_Lighting.m_Lights[i].m_N, &cl.m_Lights[i].m_N);
		Vec3ToFloat3(&pScene->m_Lighting.m_Lights[i].m_U, &cl.m_Lights[i].m_U);
		Vec3ToFloat3(&pScene->m_Lighting.m_Lights[i].m_V, &cl.m_Lights[i].m_V);
		cl.m_Lights[i].m_Area = pScene->m_Lighting.m_Lights[i].m_Area;
		cl.m_Lights[i].m_AreaPdf = pScene->m_Lighting.m_Lights[i].m_AreaPdf;
		RGBToFloat3(&pScene->m_Lighting.m_Lights[i].m_Color, &cl.m_Lights[i].m_Color);
		RGBToFloat3(&pScene->m_Lighting.m_Lights[i].m_ColorTop, &cl.m_Lights[i].m_ColorTop);
		RGBToFloat3(&pScene->m_Lighting.m_Lights[i].m_ColorMiddle, &cl.m_Lights[i].m_ColorMiddle);
		RGBToFloat3(&pScene->m_Lighting.m_Lights[i].m_ColorBottom, &cl.m_Lights[i].m_ColorBottom);
		cl.m_Lights[i].m_T = pScene->m_Lighting.m_Lights[i].m_T;
	}
	HandleCudaError(cudaMemcpyToSymbol(gLighting, &c, sizeof(CudaLighting)));

	// copy entire Scene(host mem) up to gpu.
	HandleCudaError(cudaMemcpy(pDeviceScene, pScene, sizeof(CScene), cudaMemcpyHostToDevice));
}

void Render(const int& Type, CScene& Scene, CScene* pDevScene,
	cudaFB& framebuffers,
	const cudaVolume& volumedata,
	CTiming& RenderImage, CTiming& BlurImage, CTiming& PostProcessImage, CTiming& DenoiseImage)
{
	CScene* pDScene = pDevScene;
//	CScene* pDScene = NULL;
//	HandleCudaError(cudaMalloc(&pDScene, sizeof(CScene)));
//	HandleCudaError(cudaMemcpy(pDScene, &Scene, sizeof(CScene), cudaMemcpyHostToDevice));

	//LOG_DEBUG << "CScene is " << sizeof(CScene) << " bytes";

	// find nearest intersection to set camera focal distance automatically.
	// then re-upload that data.
	if (Scene.m_Camera.m_Focus.m_Type == 0) {
		Scene.m_Camera.m_Focus.m_FocalDistance = NearestIntersection(pDScene, volumedata);
		// send m_FocalDistance back to gpu.
		CudaCamera c;
		FillCudaCamera(&Scene, c);
		HandleCudaError(cudaMemcpyToSymbol(gCamera, &c, sizeof(CudaCamera)));

		//HandleCudaError(cudaMemcpy(pDScene, &Scene, sizeof(CScene), cudaMemcpyHostToDevice));
	}

	for (int i = 0; i < Scene.m_Camera.m_Film.m_ExposureIterations; ++i) {
		CCudaTimer TmrRender;

		switch (Type)
		{
		case 0:
		{
			SingleScattering(&Scene, pDScene, volumedata, framebuffers.fb, framebuffers.randomSeeds1, framebuffers.randomSeeds2);
			break;
		}

		case 1:
		{
			//			MultipleScattering(&Scene, pDScene);
			break;
		}
		}
		RenderImage.AddDuration(TmrRender.ElapsedTime());

		// estimate just adds to accumulation buffer.
		CCudaTimer TmrPostProcess;
		Estimate(&Scene, framebuffers.fb, framebuffers.fbaccum);
		PostProcessImage.AddDuration(TmrPostProcess.ElapsedTime());

		Scene.SetNoIterations(Scene.GetNoIterations() + 1);

		const float NoIterations = Scene.GetNoIterations();
		const float InvNoIterations = 1.0f / ((NoIterations > 1.0f) ? NoIterations : 1.0f);
		HandleCudaError(cudaMemcpyToSymbol(gNoIterations, &NoIterations, sizeof(float)));
		HandleCudaError(cudaMemcpyToSymbol(gInvNoIterations, &InvNoIterations, sizeof(float)));
	}

//	HandleCudaError(cudaFree(pDScene));
//	pDScene = NULL;

}