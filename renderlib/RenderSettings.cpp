#include "Stable.h"

#include "RenderSettings.h"
#include "ImageXYZC.h"

RenderSettings::RenderSettings(void) :
	m_Camera(),
	m_DirtyFlags(),
	m_DenoiseParams(),
	m_NoIterations(0)
{
}

RenderSettings::RenderSettings(const RenderSettings& Other)
{
	*this = Other;
}

RenderSettings& RenderSettings::operator=(const RenderSettings& Other)
{
	m_Camera					= Other.m_Camera;
	m_DirtyFlags				= Other.m_DirtyFlags;
	m_DenoiseParams				= Other.m_DenoiseParams;
	m_NoIterations				= Other.m_NoIterations;
	m_RenderSettings = Other.m_RenderSettings;

	return *this;
}

void RenderSettings::initSceneFromImg(uint32_t vx, uint32_t vy, uint32_t vz, float sx, float sy, float sz)
{
	// Compute physical size
	const Vec3f PhysicalSize(Vec3f(
		sx * (float)vx,
		sy * (float)vy,
		sz * (float)vz
	));

	// Tell the camera about the volume's bounding box
	m_Camera.m_SceneBoundingBox.m_MinP = Vec3f(0.0f);
	m_Camera.m_SceneBoundingBox.m_MaxP = PhysicalSize / PhysicalSize.Max();
}

