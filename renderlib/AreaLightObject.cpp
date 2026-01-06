#include "AreaLightObject.hpp"

#include "SceneLight.h"
#include "MathUtil.h"
#include "Logging.h"

#include "serialize/docReader.h"
#include "serialize/docWriter.h"

AreaLightObject::AreaLightObject()
  : prtyObject()
{
  m_sceneLight = std::make_shared<SceneLight>();

  m_thetaUIInfo = new FloatSliderSpinnerUiInfo(&m_arealightDataObject.Theta, "Position", "Theta");
  m_thetaUIInfo->SetToolTip("Set Theta angle");
  m_thetaUIInfo->SetStatusTip("Set area light theta angle in degrees");
  m_thetaUIInfo->min = 0.0f;
  m_thetaUIInfo->max = 360.0f;
  m_thetaUIInfo->decimals = 1;
  m_thetaUIInfo->singleStep = 1.0f;
  m_thetaUIInfo->suffix = "°";
  AddProperty(m_thetaUIInfo);

  m_phiUIInfo = new FloatSliderSpinnerUiInfo(&m_arealightDataObject.Phi, "Position", "Phi");
  m_phiUIInfo->SetToolTip("Set Phi angle");
  m_phiUIInfo->SetStatusTip("Set area light phi angle in degrees");
  m_phiUIInfo->min = 0.0f;
  m_phiUIInfo->max = 180.0f;
  m_phiUIInfo->decimals = 1;
  m_phiUIInfo->singleStep = 1.0f;
  m_phiUIInfo->suffix = "°";
  AddProperty(m_phiUIInfo);

  m_sizeUIInfo = new FloatSliderSpinnerUiInfo(&m_arealightDataObject.Size, "Dimensions", "Size");
  m_sizeUIInfo->SetToolTip("Set Size");
  m_sizeUIInfo->SetStatusTip("Set area light size");
  m_sizeUIInfo->min = 0.1f;
  m_sizeUIInfo->max = 100.0f;
  m_sizeUIInfo->decimals = 2;
  m_sizeUIInfo->singleStep = 0.1f;
  AddProperty(m_sizeUIInfo);

  m_distanceUIInfo = new FloatSliderSpinnerUiInfo(&m_arealightDataObject.Distance, "Position", "Distance");
  m_distanceUIInfo->SetToolTip("Set Distance");
  m_distanceUIInfo->SetStatusTip("Set area light distance");
  m_distanceUIInfo->min = 0.1f;
  m_distanceUIInfo->max = 1000.0f;
  m_distanceUIInfo->decimals = 1;
  m_distanceUIInfo->singleStep = 1.0f;
  AddProperty(m_distanceUIInfo);

  m_intensityUIInfo =
    new ColorWithIntensityUiInfo(&m_arealightDataObject.Color, &m_arealightDataObject.Intensity, "Light", "Intensity");
  m_intensityUIInfo->SetToolTip("Set Intensity");
  m_intensityUIInfo->SetStatusTip("Set area light intensity");
  m_intensityUIInfo->min = 0.0f;
  m_intensityUIInfo->max = 1000.0f;
  m_intensityUIInfo->decimals = 1;
  m_intensityUIInfo->singleStep = 1.0f;
  AddProperty(m_intensityUIInfo);

  //   m_colorUIInfo = new ColorPickerUiInfo(&m_arealightDataObject.Color, "Light", "Color");
  //   m_colorUIInfo->SetToolTip("Set Color");
  //   m_colorUIInfo->SetStatusTip("Set area light color");
  //   AddProperty(m_colorUIInfo);

  // Add callbacks for property changes
  m_arealightDataObject.Theta.AddCallback(
    new prtyCallbackWrapper<AreaLightObject>(this, &AreaLightObject::ThetaChanged));
  m_arealightDataObject.Phi.AddCallback(new prtyCallbackWrapper<AreaLightObject>(this, &AreaLightObject::PhiChanged));
  m_arealightDataObject.Size.AddCallback(new prtyCallbackWrapper<AreaLightObject>(this, &AreaLightObject::SizeChanged));
  m_arealightDataObject.Distance.AddCallback(
    new prtyCallbackWrapper<AreaLightObject>(this, &AreaLightObject::DistanceChanged));
  m_arealightDataObject.Intensity.AddCallback(
    new prtyCallbackWrapper<AreaLightObject>(this, &AreaLightObject::IntensityChanged));
  m_arealightDataObject.Color.AddCallback(
    new prtyCallbackWrapper<AreaLightObject>(this, &AreaLightObject::ColorChanged));
}

void
AreaLightObject::updatePropsFromSceneLight()
{
  if (!m_sceneLight)
    return;

  const Light& light = m_sceneLight->m_light;

  // Convert from radians to degrees and update properties
  m_arealightDataObject.Theta.SetValue(light.m_Theta * (180.0f / PI_F));
  m_arealightDataObject.Phi.SetValue(light.m_Phi * (180.0f / PI_F));
  m_arealightDataObject.Size.SetValue(light.m_Width);
  m_arealightDataObject.Distance.SetValue(light.m_Distance);
  m_arealightDataObject.Intensity.SetValue(light.m_ColorIntensity);
  m_arealightDataObject.Color.SetValue(glm::vec4(light.m_Color, 1.0f));
}

void
AreaLightObject::updateSceneLightFromProps()
{
  if (!m_sceneLight)
    return;

  Light& light = m_sceneLight->m_light;

  // Convert from degrees to radians and update light
  light.m_Theta = m_arealightDataObject.Theta.GetValue() * (PI_F / 180.0f);
  light.m_Phi = m_arealightDataObject.Phi.GetValue() * (PI_F / 180.0f);
  light.m_Width = m_arealightDataObject.Size.GetValue();
  light.m_Distance = m_arealightDataObject.Distance.GetValue();
  light.m_ColorIntensity = m_arealightDataObject.Intensity.GetValue();

  glm::vec4 color = m_arealightDataObject.Color.GetValue();
  light.m_Color = glm::vec3(color.x, color.y, color.z);

  // Notify observers
  for (auto& observer : m_sceneLight->m_observers) {
    observer(light);
  }

  // Call dirty callback if set
  if (m_dirtyCallback) {
    m_dirtyCallback();
  }
}

void
AreaLightObject::ThetaChanged(prtyProperty* i_Property, bool i_bDirty)
{
  updateSceneLightFromProps();
}

void
AreaLightObject::PhiChanged(prtyProperty* i_Property, bool i_bDirty)
{
  updateSceneLightFromProps();
}

void
AreaLightObject::SizeChanged(prtyProperty* i_Property, bool i_bDirty)
{
  updateSceneLightFromProps();
}

void
AreaLightObject::DistanceChanged(prtyProperty* i_Property, bool i_bDirty)
{
  updateSceneLightFromProps();
}

void
AreaLightObject::IntensityChanged(prtyProperty* i_Property, bool i_bDirty)
{
  updateSceneLightFromProps();
}

void
AreaLightObject::ColorChanged(prtyProperty* i_Property, bool i_bDirty)
{
  updateSceneLightFromProps();
}

void
AreaLightObject::fromDocument(docReader* reader)
{
  // Peek at metadata
  uint32_t version = reader->peekVersion();
  std::string type = reader->peekObjectType();
  std::string name = reader->peekObjectName();

  reader->readPrty(&m_arealightDataObject.Theta);
  reader->readPrty(&m_arealightDataObject.Phi);
  reader->readPrty(&m_arealightDataObject.Size);
  reader->readPrty(&m_arealightDataObject.Distance);
  reader->readPrty(&m_arealightDataObject.Intensity);
  reader->readPrty(&m_arealightDataObject.Color);
}

void
AreaLightObject::toDocument(docWriter* writer)
{
  writer->beginObject("areaLight0", "AreaLightObject", AreaLightObject::CURRENT_VERSION);

  m_arealightDataObject.Theta.Write(*writer);
  m_arealightDataObject.Phi.Write(*writer);
  m_arealightDataObject.Size.Write(*writer);
  m_arealightDataObject.Distance.Write(*writer);
  m_arealightDataObject.Intensity.Write(*writer);
  m_arealightDataObject.Color.Write(*writer);

  writer->endObject();
}
