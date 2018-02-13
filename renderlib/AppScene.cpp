#include "AppScene.h"

#include "ImageXYZC.h"
#include "Defines.h"

void Light::Update(const CBoundingBox& BoundingBox)
{
	m_InvWidth = 1.0f / m_Width;
	m_HalfWidth = 0.5f * m_Width;
	m_InvHalfWidth = 1.0f / m_HalfWidth;
	m_InvHeight = 1.0f / m_Height;
	m_HalfHeight = 0.5f * m_Height;
	m_InvHalfHeight = 1.0f / m_HalfHeight;
	glm::vec3 bbctr = BoundingBox.GetCenter();
	m_Target = bbctr;

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
		m_P = bbctr;
		m_SkyRadius = 1000.0f * glm::length(BoundingBox.GetMaxP() - BoundingBox.GetMinP());
		m_Area = 4.0f * PI_F * powf(m_SkyRadius, 2.0f);
		m_AreaPdf = 1.0f / m_Area;
	}

	// Compute orthogonal basis frame
	m_N = glm::normalize(m_Target - m_P);
	m_U = glm::normalize(glm::cross(m_N, glm::vec3(0.0f, 1.0f, 0.0f)));
	m_V = glm::normalize(glm::cross(m_N, m_U));
}

// set up a couple of lights relative to the img's bounding box
void Scene::initSceneFromImg(std::shared_ptr<ImageXYZC> img)
{
	Light BackgroundLight;

	BackgroundLight.m_T = 1;
	float inten = 1.0f;

	float topr = 1.0f;
	float topg = 0.0f;
	float topb = 0.0f;
	float midr = 1.0f;
	float midg = 1.0f;
	float midb = 1.0f;
	float botr = 0.0f;
	float botg = 0.0f;
	float botb = 1.0f;

	BackgroundLight.m_ColorTop = inten * glm::vec3(topr, topg, topb);
	BackgroundLight.m_ColorMiddle = inten * glm::vec3(midr, midg, midb);
	BackgroundLight.m_ColorBottom = inten * glm::vec3(botr, botg, botb);

	_lighting.AddLight(BackgroundLight);

	Light AreaLight;

	AreaLight.m_T = 0;
	AreaLight.m_Theta = 0.0f / RAD_F;  // numerator is degrees
	AreaLight.m_Phi = 0.0f / RAD_F;
	AreaLight.m_Width = 1.0f;
	AreaLight.m_Height = 1.0f;
	AreaLight.m_Distance = 10.0f;
	AreaLight.m_Color = 100.0f * glm::vec3(1.0f, 1.0f, 1.0f);

	_lighting.AddLight(AreaLight);


	// point lights toward scene's bounding box

	// Compute physical size
	const glm::vec3 PhysicalSize(
		img->physicalSizeX() * (float)img->sizeX(),
		img->physicalSizeY() * (float)img->sizeY(),
		img->physicalSizeZ() * (float)img->sizeZ()
	);
	//glm::gtx::component_wise::compMax(PhysicalSize);
	float m = std::max(PhysicalSize.x, std::max(PhysicalSize.y, PhysicalSize.z));

	// Compute the volume's bounding box
	_boundingBox.m_MinP = glm::vec3(0.0f);
	_boundingBox.m_MaxP = PhysicalSize / m;

	for (int i = 0; i < _lighting.m_NoLights; ++i) {
		_lighting.m_Lights[i].Update(_boundingBox);
	}
}
