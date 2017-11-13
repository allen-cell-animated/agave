#pragma once

#include "Transport.cuh"

KERNEL void KrnlSingleScattering(CScene* pScene, cudaVolume volumedata, float* pView, unsigned int* rnd1, unsigned int* rnd2)
{
	const int X		= blockIdx.x * blockDim.x + threadIdx.x;
	const int Y		= blockIdx.y * blockDim.y + threadIdx.y;

	if (X >= gFilmWidth || Y >= gFilmHeight)
		return;
	
	// pixel offset of this thread
	int pixoffset = Y*(gFilmWidth)+(X);
	int floatoffset = pixoffset * 4;

	CRNG RNG(&rnd1[pixoffset], &rnd2[pixoffset]);

	CColorXyz Lv = SPEC_BLACK, Li = SPEC_BLACK;

	CRay Re;
	
	const Vec2f UV = Vec2f(X, Y) + RNG.Get2();

 	pScene->m_Camera.GenerateRay(UV, RNG.Get2(), Re.m_O, Re.m_D);

/*
	Lv = (CColorXyz(Re.m_D.x, Re.m_D.y, Re.m_D.z) * 0.5) + CColorXyz(1.0, 1.0, 1.0);
	pView[floatoffset] = Lv.c[0];
	pView[floatoffset + 1] = Lv.c[1];
	pView[floatoffset + 2] = Lv.c[2];
	pView[floatoffset + 3] = 1.0;
	return;
*/

	Re.m_MinT = 0.0f; 
	Re.m_MaxT = 1500000.0f;

	Vec3f Pe, Pl;
	
	CLight* pLight = NULL;
	
	if (SampleDistanceRM(Re, RNG, Pe, volumedata))
	{
		//Lv = CLR_RAD_RED;

		if (NearestLight(pScene, CRay(Re.m_O, Re.m_D, 0.0f, (Pe - Re.m_O).Length()), Li, Pl, pLight))
		{
			// set sample pixel value in frame estimate (prior to accumulation)
			pView[floatoffset] = Lv.c[0];
			pView[floatoffset + 1] = Lv.c[1];
			pView[floatoffset + 2] = Lv.c[2];
			pView[floatoffset + 3] = 1.0;
			return;
		}
		
		const float D = GetNormalizedIntensity(Pe, volumedata.volumeTexture);

		Lv += GetEmission(D).ToXYZ();

		switch (pScene->m_ShadingType)
		{
			case 0:
			{
				Lv += UniformSampleOneLight(pScene, volumedata, CVolumeShader::Brdf, D, Normalize(-Re.m_D), Pe, NormalizedGradient(Pe, volumedata.volumeTexture), RNG, true);
				//Lv = CLR_RAD_RED;
				break;
			}
		
			case 1:
			{
				Lv += 0.5f * UniformSampleOneLight(pScene, volumedata, CVolumeShader::Phase, D, Normalize(-Re.m_D), Pe, NormalizedGradient(Pe, volumedata.volumeTexture), RNG, false);
				//Lv = CLR_RAD_GREEN;
				break;
			}

			case 2:
			{
				const float GradMag = GradientMagnitude(Pe, volumedata.gradientVolumeTexture) * gIntensityInvRange;

				const float PdfBrdf = (1.0f - __expf(-pScene->m_GradientFactor * GradMag));

				CColorXyz cls;
				if (RNG.Get1() < PdfBrdf) {
					cls = UniformSampleOneLight(pScene, volumedata, CVolumeShader::Brdf, D, Normalize(-Re.m_D), Pe, NormalizedGradient(Pe, volumedata.volumeTexture), RNG, true);
				}
				else {
					cls = 0.5f * UniformSampleOneLight(pScene, volumedata, CVolumeShader::Phase, D, Normalize(-Re.m_D), Pe, NormalizedGradient(Pe, volumedata.volumeTexture), RNG, false);
				}
//				if (cls == SPEC_BLACK) {
//					Lv = CLR_RAD_RED;
//				}
//				else {
					Lv += cls;
//				}

				//Lv = CLR_RAD_BLUE;
				break;
			}
		}
	}
	else
	{
		if (NearestLight(pScene, CRay(Re.m_O, Re.m_D, 0.0f, INF_MAX), Li, Pl, pLight))
			Lv = Li;

		//Lv = CLR_RAD_GREEN;
	}

	// set sample pixel value in frame estimate (prior to accumulation)

	//float rgb[3];
	//Lv.ToRGB(rgb);
	//pView[floatoffset] = rgb[0];
	//pView[floatoffset + 1] = rgb[1];
	//pView[floatoffset + 2] = rgb[2];
	pView[floatoffset] = Lv.c[0];
	pView[floatoffset + 1] = Lv.c[1];
	pView[floatoffset + 2] = Lv.c[2];
	pView[floatoffset + 3] = 1.0;
}

void SingleScattering(CScene* pScene, CScene* pDevScene, const cudaVolume& volumedata, float* pView, unsigned int* rnd1, unsigned int* rnd2)
{
	const dim3 KernelBlock(KRNL_SS_BLOCK_W, KRNL_SS_BLOCK_H);
	const dim3 KernelGrid((int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResX() / (float)KernelBlock.x), (int)ceilf((float)pScene->m_Camera.m_Film.m_Resolution.GetResY() / (float)KernelBlock.y));
	
	KrnlSingleScattering<<<KernelGrid, KernelBlock>>>(pDevScene, volumedata, pView, rnd1, rnd2);
	HandleCudaKernelError(cudaGetLastError(), "Single Scattering");
	cudaDeviceSynchronize();
	HandleCudaKernelError(cudaGetLastError(), "Single Scattering");
}