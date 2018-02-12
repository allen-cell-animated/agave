#pragma once

#include "Flags.h"
#include "CCamera.h"
#include "DenoiseParams.cuh"

class RenderSettings
{
public:
	RenderSettings(void);
	RenderSettings(const RenderSettings& Other);
	RenderSettings& operator = (const RenderSettings& Other);

	void initCameraFromImg(uint32_t vx, uint32_t vy, uint32_t vz, float sx, float sy, float sz);

	CCamera				m_Camera;
	CFlags				m_DirtyFlags;
	CRenderSettings m_RenderSettings;
	CDenoiseParams		m_DenoiseParams;

	int GetNoIterations(void) const					{ return m_NoIterations;			}
	void SetNoIterations(const int& NoIterations)	{ m_NoIterations = NoIterations;	}

private:
	int					m_NoIterations;
};
