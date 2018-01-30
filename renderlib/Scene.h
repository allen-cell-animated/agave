#pragma once

#include "Geometry.h"
#include "Flags.h"
#include "Camera.cuh"
#include "Lighting.cuh"

class CDenoiseParams
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
	HO CDenoiseParams(void);

	HOD CDenoiseParams& operator=(const CDenoiseParams& Other)
	{
		m_Enabled			= Other.m_Enabled;
		m_Noise				= Other.m_Noise;
		m_LerpC				= Other.m_LerpC;
		m_WindowRadius		= Other.m_WindowRadius;
		m_WindowArea		= Other.m_WindowArea;
		m_InvWindowArea		= Other.m_InvWindowArea;
		m_WeightThreshold	= Other.m_WeightThreshold;
		m_LerpThreshold		= Other.m_LerpThreshold;

		return *this;
	}

	HOD void SetWindowRadius(const int& WindowRadius)
	{
		m_WindowRadius		= WindowRadius;
		m_WindowArea		= (2.0f * m_WindowRadius + 1.0f) * (2.0f * m_WindowRadius + 1.0f);
		m_InvWindowArea		= 1.0f / m_WindowArea;
	}
};

class CScene
{
public:
	CScene(void);
	CScene(const CScene& Other);
	CScene& operator = (const CScene& Other);

	HO void initSceneFromImg(uint32_t vx, uint32_t vy, uint32_t vz, float sx, float sy, float sz);

	// which channel to display.  this is "scene" display info and not "renderer settings"
	int _channel;

	CCamera				m_Camera;
	CLighting			m_Lighting;
	CResolution3D		m_Resolution;
	CFlags				m_DirtyFlags;
	Vec3f				m_Spacing;
	Vec3f				m_Scale;
	CBoundingBox		m_BoundingBox;
	//CTransferFunctions	m_TransferFunctions;
	//CRange				m_IntensityRange;
	//CRange				m_GradientMagnitudeRange;
	float				m_DensityScale;
	CDenoiseParams		m_DenoiseParams;
	float				m_Variance;
	int					m_ShadingType;
	float				m_StepSizeFactor;
	float				m_StepSizeFactorShadow;
	float				m_GradientDelta;
	float				m_GradientFactor;
	float				m_GradMagMean;

	HO int GetNoIterations(void) const					{ return m_NoIterations;			}
	HO void SetNoIterations(const int& NoIterations)	{ m_NoIterations = NoIterations;	}

private:
	int					m_NoIterations;
};
