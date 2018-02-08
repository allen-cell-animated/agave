#include "Stable.h"

#include "RenderSettings.h"
#include "ImageXYZC.h"

RenderSettings::RenderSettings(void) :
	m_Camera(),
	m_Resolution(),
	m_DirtyFlags(),
	m_Spacing(),
	m_Scale(),
	m_BoundingBox(),
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
	m_Resolution				= Other.m_Resolution;
	m_DirtyFlags				= Other.m_DirtyFlags;
	m_Spacing					= Other.m_Spacing;
	m_Scale						= Other.m_Scale;
	m_BoundingBox				= Other.m_BoundingBox;
	m_DenoiseParams				= Other.m_DenoiseParams;
	m_NoIterations				= Other.m_NoIterations;
	m_RenderSettings = Other.m_RenderSettings;

	return *this;
}

void RenderSettings::initSceneFromImg(uint32_t vx, uint32_t vy, uint32_t vz, float sx, float sy, float sz)
{
	m_Resolution.SetResX(vx);
	m_Resolution.SetResY(vy);
	m_Resolution.SetResZ(vz);
	m_Spacing.x = sx;
	m_Spacing.y = sy;
	m_Spacing.z = sz;

	// Compute physical size
	const Vec3f PhysicalSize(Vec3f(
		m_Spacing.x * (float)m_Resolution.GetResX(),
		m_Spacing.y * (float)m_Resolution.GetResY(),
		m_Spacing.z * (float)m_Resolution.GetResZ()
	));

	// Compute the volume's bounding box
	m_BoundingBox.m_MinP = Vec3f(0.0f);
	m_BoundingBox.m_MaxP = PhysicalSize / PhysicalSize.Max();

	m_Camera.m_SceneBoundingBox = m_BoundingBox;
}

