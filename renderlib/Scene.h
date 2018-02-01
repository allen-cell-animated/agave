#pragma once

#include "Geometry.h"
#include "Flags.h"
#include "CCamera.h"
#include "DenoiseParams.cuh"
//#include "Lighting.cuh"
class CLighting
{
};

class CScene
{
public:
	CScene(void);
	CScene(const CScene& Other);
	CScene& operator = (const CScene& Other);

	void initSceneFromImg(uint32_t vx, uint32_t vy, uint32_t vz, float sx, float sy, float sz);

	// which channel to display.  this is "scene" display info and not "renderer settings"
	//int _channel;

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
	CRenderSettings m_RenderSettings;
	//float				m_DensityScale;
	CDenoiseParams		m_DenoiseParams;
	//float				m_Variance;
	//int					m_ShadingType;
	//float				m_StepSizeFactor;
	//float				m_StepSizeFactorShadow;
	//float				m_GradientDelta;
	//float				m_GradientFactor;
	//float				m_GradMagMean;

	int GetNoIterations(void) const					{ return m_NoIterations;			}
	void SetNoIterations(const int& NoIterations)	{ m_NoIterations = NoIterations;	}

private:
	int					m_NoIterations;
};
