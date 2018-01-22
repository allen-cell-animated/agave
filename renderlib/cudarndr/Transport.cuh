#pragma once

#include "Shader.cuh"
#include "RayMarching.cuh"

DEV CColorXyz EstimateDirectLight(const CudaLighting& lighting, const cudaVolume& volumedata, const CVolumeShader::EType& Type, const float& Density, int ch, const CudaLight& Light, CLightingSample& LS, const Vec3f& Wo, const Vec3f& Pe, const Vec3f& N, CRNG& RNG)
{
	CColorXyz Ld = SPEC_BLACK, Li = SPEC_BLACK, F = SPEC_BLACK;
	
	CVolumeShader Shader(Type, N, Wo, GetDiffuseN(Density, volumedata, ch).ToXYZ(), GetSpecularN(Density, volumedata, ch).ToXYZ(), 2.5f/*pScene->m_IOR*/, GetRoughnessN(Density, volumedata, ch));
	
	CRay Rl; 

	float LightPdf = 1.0f, ShaderPdf = 1.0f;
	
	Vec3f Wi, P, Pl;

 	Li = Light.SampleL(Pe, Rl, LightPdf, LS);
	
	const CudaLight* pLight = NULL;

	Wi = -Rl.m_D; 

	F = Shader.F(Wo, Wi); 

	ShaderPdf = Shader.Pdf(Wo, Wi);

	if (!Li.IsBlack() && ShaderPdf > 0.0f && LightPdf > 0.0f && !FreePathRM(Rl, RNG, volumedata))
	{
		const float WeightMIS = PowerHeuristic(1.0f, LightPdf, 1.0f, ShaderPdf);
		
		if (Type == CVolumeShader::Brdf)
			Ld += F * Li * AbsDot(Wi, N) * WeightMIS / LightPdf;

		if (Type == CVolumeShader::Phase)
			Ld += F * Li * WeightMIS / LightPdf;
	}

	F = Shader.SampleF(Wo, Wi, ShaderPdf, LS.m_BsdfSample);

	if (!F.IsBlack() && ShaderPdf > 0.0f)
	{
		int n = NearestLight(lighting, CRay(Pe, Wi, 0.0f), Li, Pl, &LightPdf);
		if (n > -1)
		{
			pLight = &gLighting.m_Lights[n];
			LightPdf = pLight->Pdf(Pe, Wi);

			if ((LightPdf > 0.0f) &&
				!Li.IsBlack()) {
				CRay rr(Pl, Normalize(Pe - Pl), 0.0f, (Pe - Pl).Length());
				if (!FreePathRM(rr, RNG, volumedata))
				{
					const float WeightMIS = PowerHeuristic(1.0f, ShaderPdf, 1.0f, LightPdf);

					if (Type == CVolumeShader::Brdf)
						Ld += F * Li * AbsDot(Wi, N) * WeightMIS / ShaderPdf;

					if (Type == CVolumeShader::Phase)
						Ld += F * Li * WeightMIS / ShaderPdf;
				}

			}
		}
	}

	return Ld;
}

DEV CColorXyz UniformSampleOneLight(const CudaLighting& lighting, const cudaVolume& volumedata, const CVolumeShader::EType& Type, const float& Density, int ch, const Vec3f& Wo, const Vec3f& Pe, const Vec3f& N, CRNG& RNG, const bool& Brdf)
{
	const int NumLights = lighting.m_NoLights;

 	if (NumLights == 0)
 		return SPEC_BLACK;

	CLightingSample LS;

	LS.LargeStep(RNG);

	const int WhichLight = (int)floorf(LS.m_LightNum * (float)NumLights);

	const CudaLight& Light = lighting.m_Lights[WhichLight];

	return (float)NumLights * EstimateDirectLight(lighting, volumedata, Type, Density, ch, Light, LS, Wo, Pe, N, RNG);
}