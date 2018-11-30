#pragma once

#include "Geometry.h"

#include "MonteCarlo.cuh"
#include "Sample.cuh"

class CLambertian
{
public:
	HOD CLambertian(const CColorXyz& Kd)
	{
		m_Kd = Kd;
	}

	HOD ~CLambertian(void)
	{
	}

	HOD CColorXyz F(const float3& Wo, const float3& Wi)
	{
		return m_Kd * INV_PI_F;
	}

	HOD CColorXyz SampleF(const float3& Wo, float3& Wi, float& Pdf, const float2& U)
	{
		Wi = CosineWeightedHemisphere(U);

		if (Wo.z < 0.0f)
			Wi.z *= -1.0f;

		Pdf = this->Pdf(Wo, Wi);

		return this->F(Wo, Wi);
	}

	HOD float Pdf(const float3& Wo, const float3& Wi)
	{
		return SameHemisphere(Wo, Wi) ? AbsCosTheta(Wi) * INV_PI_F : 0.0f;
	}

	CColorXyz	m_Kd;
};

HOD inline CColorXyz FrDiel(float cosi, float cost, const CColorXyz &etai, const CColorXyz &etat)
{
	CColorXyz Rparl = ((etat * cosi) - (etai * cost)) / ((etat * cosi) + (etai * cost));
	CColorXyz Rperp = ((etai * cosi) - (etat * cost)) / ((etai * cosi) + (etat * cost));
	return (Rparl*Rparl + Rperp*Rperp) / 2.f;
}

class CFresnel
{
public:
	HOD CFresnel(float ei, float et) :
	  eta_i(ei),
		  eta_t(et)
	  {
	  }

	  HOD  ~CFresnel(void)
	  {
	  }

	  HOD CColorXyz Evaluate(float cosi)
	  {
		  // Compute Fresnel reflectance for dielectric
		  cosi = Clamp(cosi, -1.0f, 1.0f);

		  // Compute indices of refraction for dielectric
		  bool entering = cosi > 0.0f;
		  float ei = eta_i, et = eta_t;

		  if (!entering)
			  swap(ei, et);

		  // Compute _sint_ using Snell's law
		  float sint = ei/et * sqrtf(max(0.f, 1.f - cosi*cosi));

		  if (sint >= 1.0f)
		  {
			  // Handle total internal reflection
			  return 1.0f;
		  }
		  else
		  {
			  float cost = sqrtf(max(0.f, 1.0f - sint * sint));
			  return FrDiel(fabsf(cosi), cost, ei, et);
		  }
	  }

	  float eta_i, eta_t;
};

class CBlinn
{
public:
	HOD CBlinn(const float& Exponent) :
	  m_Exponent(Exponent)
	  {
	  }

	  HOD ~CBlinn(void)
	  {
	  }

	  HOD void SampleF(const float3& Wo, float3& Wi, float& Pdf, const float2& U)
	  {
		  // Compute sampled half-angle vector $\wh$ for Blinn distribution
		  float costheta = powf(U.x, 1.f / (m_Exponent+1));
		  float sintheta = sqrtf(max(0.f, 1.f - costheta*costheta));
		  float phi = U.y * 2.f * PI_F;

		  float3 wh = SphericalDirection(sintheta, costheta, phi);

		  if (!SameHemisphere(Wo, wh))
			  wh = -wh;

		  // Compute incident direction by reflecting about $\wh$
		  Wi = (-1.0*Wo) + 2.f * dot(Wo, wh) * wh;

		  // Compute PDF for $\wi$ from Blinn distribution
		  float blinn_pdf = ((m_Exponent + 1.f) * powf(costheta, m_Exponent)) / (2.f * PI_F * 4.f * dot(Wo, wh));

		  if (dot(Wo, wh) <= 0.f)
			  blinn_pdf = 0.f;

		  Pdf = blinn_pdf;
	  }

	  HOD float Pdf(const float3& Wo, const float3& Wi)
	  {
		  float3 wh = normalize(Wo + Wi);

		  float costheta = AbsCosTheta(wh);
		  // Compute PDF for $\wi$ from Blinn distribution
		  float blinn_pdf = ((m_Exponent + 1.f) * powf(costheta, m_Exponent)) / (2.f * PI_F * 4.f * dot(Wo, wh));

		  if (dot(Wo, wh) <= 0.0f)
			  blinn_pdf = 0.0f;

		  return blinn_pdf;
	  }

	  HOD float D(const float3& wh)
	  {
		  float costhetah = AbsCosTheta(wh);
		  return (m_Exponent+2) * INV_TWO_PI_F * powf(costhetah, m_Exponent);
	  }

	  float	m_Exponent;
};

class CMicrofacet
{
public:
	HOD CMicrofacet(const CColorXyz& Reflectance, const float& Ior, const float& Exponent) :
	  m_R(Reflectance),
		  m_Fresnel(Ior, 1.0f),
		  m_Blinn(Exponent)
	  {
	  }

	  HOD ~CMicrofacet(void)
	  {
	  }

	  HOD CColorXyz F(const float3& wo, const float3& wi)
	  {
		  float cosThetaO = AbsCosTheta(wo);
		  float cosThetaI = AbsCosTheta(wi);

		  if (cosThetaI == 0.f || cosThetaO == 0.f)
			  return SPEC_BLACK;

		  float3 wh = wi + wo;

		  if (wh.x == 0. && wh.y == 0. && wh.z == 0.)
			  return SPEC_BLACK;

		  wh = normalize(wh);
		  float cosThetaH = dot(wi, wh);

		  CColorXyz F = SPEC_WHITE;//m_Fresnel.Evaluate(cosThetaH);

		  return m_R * m_Blinn.D(wh) * G(wo, wi, wh) * F / (4.f * cosThetaI * cosThetaO);
	  }

	  HOD CColorXyz SampleF(const float3& wo, float3& wi, float& Pdf, const float2& U)
	  {
		  m_Blinn.SampleF(wo, wi, Pdf, U);

		  if (!SameHemisphere(wo, wi))
			  return SPEC_BLACK;

		  return this->F(wo, wi);
	  }

	  HOD float Pdf(const float3& wo, const float3& wi)
	  {
		  if (!SameHemisphere(wo, wi))
			  return 0.0f;

		  return m_Blinn.Pdf(wo, wi);
	  }

	  HOD float G(const float3& wo, const float3& wi, const float3& wh)
	  {
		  float NdotWh = AbsCosTheta(wh);
		  float NdotWo = AbsCosTheta(wo);
		  float NdotWi = AbsCosTheta(wi);
		  float WOdotWh = AbsDot(wo, wh);

		  return min(1.f, min((2.f * NdotWh * NdotWo / WOdotWh), (2.f * NdotWh * NdotWi / WOdotWh)));
	  }

	  CColorXyz		m_R;
	  CFresnel		m_Fresnel;
	  CBlinn		m_Blinn;

};

class CIsotropicPhase
{
public:
	HOD CIsotropicPhase(const CColorXyz& Kd) :
		m_Kd(Kd)
	{
	}

	HOD ~CIsotropicPhase(void)
	{
	}

	HOD CColorXyz F(const float3& Wo, const float3& Wi)
	{
		return m_Kd * INV_PI_F;
	}

	HOD CColorXyz SampleF(const float3& Wo, float3& Wi, float& Pdf, const float2& U)
	{
		Wi	= UniformSampleSphere(U);
		Pdf	= this->Pdf(Wo, Wi);

		return F(Wo, Wi);
	}

	HOD float Pdf(const float3& Wo, const float3& Wi)
	{
		return INV_4_PI_F;
	}

	CColorXyz	m_Kd;
};

class CBRDF
{
public:
	HOD CBRDF(const float3& N, const float3& Wo, const CColorXyz& Kd, const CColorXyz& Ks, const float& Ior, const float& Exponent) :
		m_Lambertian(Kd),
		m_Microfacet(Ks, Ior, Exponent),
		m_Nn(N),
		m_Nu(normalize(cross(N, Wo))),
		m_Nv(normalize(cross(N, m_Nu)))
	{
	}

	HOD ~CBRDF(void)
	{
	}

	HOD float3 WorldToLocal(const float3& W)
	{
		return make_float3(dot(W, m_Nu), dot(W, m_Nv), dot(W, m_Nn));
	}

	HOD float3 LocalToWorld(const float3& W)
	{
		return make_float3(	m_Nu.x * W.x + m_Nv.x * W.y + m_Nn.x * W.z,
						m_Nu.y * W.x + m_Nv.y * W.y + m_Nn.y * W.z,
						m_Nu.z * W.x + m_Nv.z * W.y + m_Nn.z * W.z);
	}

	HOD CColorXyz F(const float3& Wo, const float3& Wi)
	{
		const float3 Wol = WorldToLocal(Wo);
		const float3 Wil = WorldToLocal(Wi);

		CColorXyz R;

		R += m_Lambertian.F(Wol, Wil);
		R += m_Microfacet.F(Wol, Wil);

		return R;
	}

	HOD CColorXyz SampleF(const float3& Wo, float3& Wi, float& Pdf, const CBrdfSample& S)
	{
		const float3 Wol = WorldToLocal(Wo);
		float3 Wil;

		CColorXyz R;

		if (S.m_Component <= 0.5f)
		{
			m_Lambertian.SampleF(Wol, Wil, Pdf, S.m_Dir);
		}
		else
		{
			m_Microfacet.SampleF(Wol, Wil, Pdf, S.m_Dir);
		}

		Pdf += m_Lambertian.Pdf(Wol, Wil);
		Pdf += m_Microfacet.Pdf(Wol, Wil);

		R += m_Lambertian.F(Wol, Wil);
		R += m_Microfacet.F(Wol, Wil);

		Wi = LocalToWorld(Wil);

		return R;
	}

	HOD float Pdf(const float3& Wo, const float3& Wi)
	{
		const float3 Wol = WorldToLocal(Wo);
		const float3 Wil = WorldToLocal(Wi);

		float Pdf = 0.0f;

		Pdf += m_Lambertian.Pdf(Wol, Wil);
		Pdf += m_Microfacet.Pdf(Wol, Wil);

		return Pdf;
	}

	float3			m_Nn;
	float3			m_Nu;
	float3			m_Nv;
	CLambertian		m_Lambertian;
	CMicrofacet		m_Microfacet;
};

class CVolumeShader
{
public:
	enum EType
	{
		Brdf,
		Phase
	};

	HOD CVolumeShader(const EType& Type, const float3& N, const float3& Wo, const CColorXyz& Kd, const CColorXyz& Ks, const float& Ior, const float& Exponent) :
		m_Type(Type),
		m_Brdf(N, Wo, Kd, Ks, Ior, Exponent),
		m_IsotropicPhase(Kd)
	{
	}

	HOD ~CVolumeShader(void)
	{
	}

	HOD CColorXyz F(const float3& Wo, const float3& Wi)
	{
		switch (m_Type)
		{
			case Brdf:
				return m_Brdf.F(Wo, Wi);

			case Phase:
				return m_IsotropicPhase.F(Wo, Wi);
		}

		return 1.0f;
	}

	HOD CColorXyz SampleF(const float3& Wo, float3& Wi, float& Pdf, const CBrdfSample& S)
	{
		switch (m_Type)
		{
			case Brdf:
				return m_Brdf.SampleF(Wo, Wi, Pdf, S);

			case Phase:
				return m_IsotropicPhase.SampleF(Wo, Wi, Pdf, S.m_Dir);
		}

		return CColorXyz(0.0f);
	}

	HOD float Pdf(const float3& Wo, const float3& Wi)
	{
		switch (m_Type)
		{
			case Brdf:
				return m_Brdf.Pdf(Wo, Wi);

			case Phase:
				return m_IsotropicPhase.Pdf(Wo, Wi);
		}

		return 1.0f;
	}

	EType				m_Type;
	CBRDF				m_Brdf;
	CIsotropicPhase		m_IsotropicPhase;
};