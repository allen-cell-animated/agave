#pragma once

#include "glm.h"
#include "Geometry.h"

#include <memory>

class ImageXYZC;

class DenoiseParams
{
public:
	bool		m_Enabled;
	float		m_Noise;
	float		m_LerpC;
	int		m_WindowRadius;
	float		m_WindowArea;
	float		m_InvWindowArea;
	float		m_WeightThreshold;
	float		m_LerpThreshold;

public:
	DenoiseParams(void)
	{
		/*
		m_Enabled			= true;
		m_Noise				= 1.0f / (0.32f * 0.32f);
		m_LerpC				= 0.2f;
		m_WindowRadius		= 2.0f;
		m_WindowArea		= (2.0f * m_WindowRadius + 1.0f) * (2.0f * m_WindowRadius + 1.0f);
		m_InvWindowArea		= 1.0f / m_WindowArea;
		m_WeightThreshold	= 0.02f;
		m_LerpThreshold		= 0.79f;
		*/


		m_Enabled = true;
		m_Noise = 0.05f;//0.32f * 0.32f;// / (0.1f * 0.1f);
		m_LerpC = 0.01f;
		m_WindowRadius = 6;
		m_WindowArea = (2.0f * m_WindowRadius + 1.0f) * (2.0f * m_WindowRadius + 1.0f);
		m_InvWindowArea = 1.0f / m_WindowArea;
		m_WeightThreshold = 0.1f;
		m_LerpThreshold = 0.0f;
		/**/
	}

	DenoiseParams& operator=(const DenoiseParams& Other)
	{
		m_Enabled = Other.m_Enabled;
		m_Noise = Other.m_Noise;
		m_LerpC = Other.m_LerpC;
		m_WindowRadius = Other.m_WindowRadius;
		m_WindowArea = Other.m_WindowArea;
		m_InvWindowArea = Other.m_InvWindowArea;
		m_WeightThreshold = Other.m_WeightThreshold;
		m_LerpThreshold = Other.m_LerpThreshold;

		return *this;
	}

	void SetWindowRadius(const int& WindowRadius)
	{
		m_WindowRadius = WindowRadius;
		m_WindowArea = (2.0f * m_WindowRadius + 1.0f) * (2.0f * m_WindowRadius + 1.0f);
		m_InvWindowArea = 1.0f / m_WindowArea;
	}
};

class RenderParams {
	DenoiseParams		m_DenoiseParams;
	float				m_StepSizeFactor;
	float				m_StepSizeFactorShadow;
};


#define MAX_CPU_CHANNELS 32
struct VolumeDisplay {
	float m_DensityScale;
    float m_GradientFactor;
	int m_ShadingType;

    // channels enabled/disabled
    // per channel colors
	float diffuse[MAX_CPU_CHANNELS * 3];
	float specular[MAX_CPU_CHANNELS * 3];
	float emissive[MAX_CPU_CHANNELS * 3];
	float roughness[MAX_CPU_CHANNELS];
	bool enabled[MAX_CPU_CHANNELS];
};

class Light {
public:
	float			m_Theta;
	float			m_Phi;
	float			m_Width;
	float			m_InvWidth;
	float			m_HalfWidth;
	float			m_InvHalfWidth;
	float			m_Height;
	float			m_InvHeight;
	float			m_HalfHeight;
	float			m_InvHalfHeight;
	float			m_Distance;
	float			m_SkyRadius;
	glm::vec3		m_P;
	glm::vec3		m_Target;
	glm::vec3		m_N;
	glm::vec3		m_U;
	glm::vec3		m_V;
	float			m_Area;
	float			m_AreaPdf;
	glm::vec3	m_Color;
	glm::vec3	m_ColorTop;
	glm::vec3	m_ColorMiddle;
	glm::vec3	m_ColorBottom;
	int				m_T;

	Light(void) :
		m_Theta(0.0f),
		m_Phi(0.0f),
		m_Width(1.0f),
		m_InvWidth(1.0f / m_Width),
		m_HalfWidth(0.5f * m_Width),
		m_InvHalfWidth(1.0f / m_HalfWidth),
		m_Height(1.0f),
		m_InvHeight(1.0f / m_Height),
		m_HalfHeight(0.5f * m_Height),
		m_InvHalfHeight(1.0f / m_HalfHeight),
		m_Distance(1.0f),
		m_SkyRadius(100.0f),
		m_P(1.0f, 1.0f, 1.0f),
		m_Target(0.0f, 0.0f, 0.0f),
		m_N(1.0f, 0.0f, 0.0f),
		m_U(1.0f, 0.0f, 0.0f),
		m_V(1.0f, 0.0f, 0.0f),
		m_Area(m_Width * m_Height),
		m_AreaPdf(1.0f / m_Area),
		m_Color(10.0f),
		m_ColorTop(10.0f),
		m_ColorMiddle(10.0f),
		m_ColorBottom(10.0f),
		m_T(0)
	{
	}

	Light& operator=(const Light& Other)
	{
		m_Theta				= Other.m_Theta;
		m_Phi				= Other.m_Phi;
		m_Width				= Other.m_Width;
		m_InvWidth			= Other.m_InvWidth;
		m_HalfWidth			= Other.m_HalfWidth;
		m_InvHalfWidth		= Other.m_InvHalfWidth;
		m_Height			= Other.m_Height;
		m_InvHeight			= Other.m_InvHeight;
		m_HalfHeight		= Other.m_HalfHeight;
		m_InvHalfHeight		= Other.m_InvHalfHeight;
		m_Distance			= Other.m_Distance;
		m_SkyRadius			= Other.m_SkyRadius;
		m_P					= Other.m_P;
		m_Target			= Other.m_Target;
		m_N					= Other.m_N;
		m_U					= Other.m_U;
		m_V					= Other.m_V;
		m_Area				= Other.m_Area;
		m_AreaPdf			= Other.m_AreaPdf;
		m_Color				= Other.m_Color;
		m_ColorTop			= Other.m_ColorTop;
		m_ColorMiddle		= Other.m_ColorMiddle;
		m_ColorBottom		= Other.m_ColorBottom;
		m_T					= Other.m_T;

		return *this;
	}

	void Update(const CBoundingBox& BoundingBox);

};

#define MAX_NO_LIGHTS 4
class Lighting {
public:
	Lighting(void) :
		m_NoLights(0)
	{
	}

	Lighting& operator=(const Lighting& Other)
	{
		for (int i = 0; i < MAX_NO_LIGHTS; i++)
		{
			m_Lights[i] = Other.m_Lights[i];
		}

		m_NoLights = Other.m_NoLights;

		return *this;
	}

	void AddLight(const Light& Light)
	{
 		if (m_NoLights >= MAX_NO_LIGHTS)
 			return;

		m_Lights[m_NoLights] = Light;

		m_NoLights = m_NoLights + 1;
	}

	void Reset(void)
	{
		m_NoLights = 0;
		//memset(m_Lights, 0 , MAX_NO_LIGHTS * sizeof(CLight));
	}

	Light		m_Lights[MAX_NO_LIGHTS];
	int			m_NoLights;
};

class Scene {
public:
	// one single volume, for now...!
    std::shared_ptr<ImageXYZC> _volume;
	// appearance settings for a volume
    VolumeDisplay _material;

    Lighting _lighting;

	CBoundingBox _boundingBox;
	void initSceneFromImg(std::shared_ptr<ImageXYZC> img);
};
