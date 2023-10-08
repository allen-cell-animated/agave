#include "AppScene.h"

#include "Defines.h"
#include "ImageXYZC.h"

#include <glm/gtx/color_space.hpp>

void
Light::Update(const CBoundingBox& BoundingBox)
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
  m_P.x = m_Distance * sinf(m_Phi) * sinf(m_Theta);
  m_P.z = m_Distance * sinf(m_Phi) * cosf(m_Theta);
  m_P.y = m_Distance * cosf(m_Phi);

  m_P += m_Target;

  // Determine area
  if (m_T == 0) {
    m_Area = m_Width * m_Height;
    m_AreaPdf = 1.0f / m_Area;
  }

  if (m_T == 1) {
    m_P = bbctr;
    // shift by nonzero amount
    m_Target = m_P + glm::vec3(0.0, 0.0, 1.0);
    m_SkyRadius = 1000.0f * glm::length(BoundingBox.GetMaxP() - BoundingBox.GetMinP());
    m_Area = 4.0f * PI_F * powf(m_SkyRadius, 2.0f);
    m_AreaPdf = 1.0f / m_Area;
  }

  // after target and p are set...
  updateBasisFrame();
}

void
Light::updateBasisFrame()
{
  // Compute orthogonal basis frame
  m_N = glm::normalize(m_Target - m_P);
  // if N and "up" are parallel, then just choose a different "up"
  if (m_N.y == 1.0f || m_N.y == -1.0f) {
    m_U = glm::normalize(glm::cross(m_N, glm::vec3(1.0f, 0.0f, 0.0f)));
  } else {
    // standard "up" vector
    m_U = glm::normalize(glm::cross(m_N, glm::vec3(0.0f, 1.0f, 0.0f)));
  }
  m_V = glm::normalize(glm::cross(m_N, m_U));
}

void
Scene::initLights()
{
  Light BackgroundLight;

  BackgroundLight.m_T = 1;
  float inten = 1.0f;

  float topr = 0.5f;
  float topg = 0.5f;
  float topb = 0.5f;
  float midr = 0.5f;
  float midg = 0.5f;
  float midb = 0.5f;
  float botr = 0.5f;
  float botg = 0.5f;
  float botb = 0.5f;

  BackgroundLight.m_ColorTop = inten * glm::vec3(topr, topg, topb);
  BackgroundLight.m_ColorMiddle = inten * glm::vec3(midr, midg, midb);
  BackgroundLight.m_ColorBottom = inten * glm::vec3(botr, botg, botb);

  m_lighting.AddLight(BackgroundLight);

  Light AreaLight;

  AreaLight.m_T = 0;
  AreaLight.m_Theta = 0.0f;
  AreaLight.m_Phi = HALF_PI_F;
  AreaLight.m_Width = 0.15f;
  AreaLight.m_Height = 0.15f;
  AreaLight.m_Distance = 1.5f;
  AreaLight.m_Color = 10.0f * glm::vec3(1.0f, 1.0f, 1.0f);

  m_lighting.AddLight(AreaLight);
}

inline std::vector<float>
rndColors(int count)
{
  std::vector<float> colors;
  // colors.push_back(QColor(255, 0, 255));
  colors.push_back(1.0);
  colors.push_back(0.0);
  colors.push_back(1.0);

  // colors.push_back(QColor(255, 255, 255));
  colors.push_back(1.0);
  colors.push_back(1.0);
  colors.push_back(1.0);

  // colors.push_back(QColor(0, 255, 255));
  colors.push_back(0.0);
  colors.push_back(1.0);
  colors.push_back(1.0);

  float currentHue = 0.0;
  for (int i = 0; i < count; i++) {
    glm::vec3 rgb = glm::rgbColor(glm::vec3(currentHue * 360.0f, 1.0f, 0.5f));
    colors.push_back(rgb.x);
    colors.push_back(rgb.y);
    colors.push_back(rgb.z);

    // QColor c = QColor::fromHslF(currentHue, 1.0, 0.5);

    // colors.push_back(c.redF());
    // colors.push_back(c.greenF());
    // colors.push_back(c.blueF());

    // this add plus the mod is simulating a jump that will cycle
    // in a pseudo-random fashion
    currentHue += 0.618033988749895f;
    currentHue = std::fmod(currentHue, 1.0f);
  }
  return colors;
}

// set up a couple of lights relative to the img's bounding box
void
Scene::initSceneFromImg(std::shared_ptr<ImageXYZC> img)
{
  std::vector<float> colors = rndColors(img->sizeC());

  for (uint32_t i = 0; i < img->sizeC(); ++i) {
    // enable first 3 channels!
    m_material.m_enabled[i] = (i < ImageXYZC::FIRST_N_CHANNELS);

    m_material.m_diffuse[i * 3] = colors[i * 3];
    m_material.m_diffuse[i * 3 + 1] = colors[i * 3 + 1];
    m_material.m_diffuse[i * 3 + 2] = colors[i * 3 + 2];

    m_material.m_specular[i * 3] = 0.0;
    m_material.m_specular[i * 3 + 1] = 0.0;
    m_material.m_specular[i * 3 + 2] = 0.0;

    m_material.m_emissive[i * 3] = 0.0;
    m_material.m_emissive[i * 3 + 1] = 0.0;
    m_material.m_emissive[i * 3 + 2] = 0.0;

    m_material.m_opacity[i] = 1.0;
    m_material.m_roughness[i] = 1.0;
  }

  initBoundsFromImg(img);
}

void
Scene::initBoundsFromImg(std::shared_ptr<ImageXYZC> img)
{
  glm::vec3 dim = img->getDimensions();

  initBounds(CBoundingBox(glm::vec3(0.0f), dim));
}

void
Scene::initBounds(const CBoundingBox& bb)
{
  // Compute the volume's bounding box
  m_boundingBox.m_MinP = bb.GetMinP();
  m_boundingBox.m_MaxP = bb.GetMaxP();

  // point lights toward scene's bounding box
  for (int i = 0; i < m_lighting.m_NoLights; ++i) {
    m_lighting.m_Lights[i].Update(m_boundingBox);
  }
}

void
Scene::getFirst4EnabledChannels(uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3) const
{
  uint32_t ch[4] = { 0u, 0u, 0u, 0u };
  uint32_t activeChannel = 0u;
  uint32_t NC = m_volume->sizeC();
  for (uint32_t i = 0; i < NC; ++i) {
    if (m_material.m_enabled[i] && activeChannel < 4u) {
      ch[activeChannel] = i;
      activeChannel++;
    }
  }
  c0 = ch[0];
  c1 = ch[1];
  c2 = ch[2];
  c3 = ch[3];
}
