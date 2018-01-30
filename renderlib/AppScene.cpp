#include "AppScene.h"

#include "Geometry.h"

void Light::Update(const CBoundingBox& BoundingBox)
{
	m_InvWidth = 1.0f / m_Width;
	m_HalfWidth = 0.5f * m_Width;
	m_InvHalfWidth = 1.0f / m_HalfWidth;
	m_InvHeight = 1.0f / m_Height;
	m_HalfHeight = 0.5f * m_Height;
	m_InvHalfHeight = 1.0f / m_HalfHeight;
	Vec3f bbctr = BoundingBox.GetCenter();
	m_Target = glm::vec3(bbctr.x, bbctr.y, bbctr.z);

	// Determine light position
	m_P.x = m_Distance * cosf(m_Phi) * sinf(m_Theta);
	m_P.z = m_Distance * cosf(m_Phi) * cosf(m_Theta);
	m_P.y = m_Distance * sinf(m_Phi);

	m_P += m_Target;

	// Determine area
	if (m_T == 0)
	{
		m_Area = m_Width * m_Height;
		m_AreaPdf = 1.0f / m_Area;
	}

	if (m_T == 1)
	{
		m_P = glm::vec3(bbctr.x, bbctr.y, bbctr.z);
		m_SkyRadius = 1000.0f * (BoundingBox.GetMaxP() - BoundingBox.GetMinP()).Length();
		m_Area = 4.0f * PI_F * powf(m_SkyRadius, 2.0f);
		m_AreaPdf = 1.0f / m_Area;
	}

	// Compute orthogonal basis frame
	m_N = glm::normalize(m_Target - m_P);
	m_U = glm::normalize(glm::cross(m_N, glm::vec3(0.0f, 1.0f, 0.0f)));
	m_V = glm::normalize(glm::cross(m_N, m_U));
}

void Scene::initSceneFromImg(uint32_t vx, uint32_t vy, uint32_t vz, float sx, float sy, float sz)
{
	//Log("Spacing: " + FormatSize(gScene.m_Spacing, 2), "grid");

	// Compute physical size
	const Vec3f PhysicalSize(Vec3f(
		sx * (float)vx,
		sy * (float)vy,
		sz * (float)vz
	));

	// Compute the volume's bounding box
	CBoundingBox bb;
	bb.m_MinP = Vec3f(0.0f);
	bb.m_MaxP = PhysicalSize / PhysicalSize.Max();

//	m_Camera.m_SceneBoundingBox = m_BoundingBox;

	for (int i = 0; i < _lighting.m_NoLights; ++i) {
		_lighting.m_Lights[i].Update(bb);
	}
}
