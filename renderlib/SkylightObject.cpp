#include "SkylightObject.hpp"

#include "SceneLight.h"
#include "Logging.h"

SkylightObject::SkylightObject()
  : prtyObject()
{
  m_topIntensityUIInfo = new FloatSliderSpinnerUiInfo(&m_skylightDataObject.TopIntensity, "Intensity", "Top Intensity");
  m_topIntensityUIInfo->SetToolTip("Set Top Intensity");
  m_topIntensityUIInfo->SetStatusTip("Set skylight top intensity");
  m_topIntensityUIInfo->min = 0.0f;
  m_topIntensityUIInfo->max = 10.0f;
  m_topIntensityUIInfo->decimals = 2;
  m_topIntensityUIInfo->singleStep = 0.1f;
  AddProperty(m_topIntensityUIInfo);

  m_topColorUIInfo = new ColorPickerUiInfo(&m_skylightDataObject.TopColor, "Color", "Top Color");
  m_topColorUIInfo->SetToolTip("Set Top Color");
  m_topColorUIInfo->SetStatusTip("Set skylight top color");
  AddProperty(m_topColorUIInfo);

  m_middleIntensityUIInfo =
    new FloatSliderSpinnerUiInfo(&m_skylightDataObject.MiddleIntensity, "Intensity", "Middle Intensity");
  m_middleIntensityUIInfo->SetToolTip("Set Middle Intensity");
  m_middleIntensityUIInfo->SetStatusTip("Set skylight middle intensity");
  m_middleIntensityUIInfo->min = 0.0f;
  m_middleIntensityUIInfo->max = 10.0f;
  m_middleIntensityUIInfo->decimals = 2;
  m_middleIntensityUIInfo->singleStep = 0.1f;
  AddProperty(m_middleIntensityUIInfo);

  m_middleColorUIInfo = new ColorPickerUiInfo(&m_skylightDataObject.MiddleColor, "Color", "Middle Color");
  m_middleColorUIInfo->SetToolTip("Set Middle Color");
  m_middleColorUIInfo->SetStatusTip("Set skylight middle color");
  AddProperty(m_middleColorUIInfo);

  m_bottomIntensityUIInfo =
    new FloatSliderSpinnerUiInfo(&m_skylightDataObject.BottomIntensity, "Intensity", "Bottom Intensity");
  m_bottomIntensityUIInfo->SetToolTip("Set Bottom Intensity");
  m_bottomIntensityUIInfo->SetStatusTip("Set skylight bottom intensity");
  m_bottomIntensityUIInfo->min = 0.0f;
  m_bottomIntensityUIInfo->max = 10.0f;
  m_bottomIntensityUIInfo->decimals = 2;
  m_bottomIntensityUIInfo->singleStep = 0.1f;
  AddProperty(m_bottomIntensityUIInfo);

  m_bottomColorUIInfo = new ColorPickerUiInfo(&m_skylightDataObject.BottomColor, "Color", "Bottom Color");
  m_bottomColorUIInfo->SetToolTip("Set Bottom Color");
  m_bottomColorUIInfo->SetStatusTip("Set skylight bottom color");
  AddProperty(m_bottomColorUIInfo);

  m_skylightDataObject.TopIntensity.AddCallback(
    new prtyCallbackWrapper<SkylightObject>(this, &SkylightObject::TopIntensityChanged));
  m_skylightDataObject.TopColor.AddCallback(
    new prtyCallbackWrapper<SkylightObject>(this, &SkylightObject::TopColorChanged));
  m_skylightDataObject.MiddleIntensity.AddCallback(
    new prtyCallbackWrapper<SkylightObject>(this, &SkylightObject::MiddleIntensityChanged));
  m_skylightDataObject.MiddleColor.AddCallback(
    new prtyCallbackWrapper<SkylightObject>(this, &SkylightObject::MiddleColorChanged));
  m_skylightDataObject.BottomIntensity.AddCallback(
    new prtyCallbackWrapper<SkylightObject>(this, &SkylightObject::BottomIntensityChanged));
  m_skylightDataObject.BottomColor.AddCallback(
    new prtyCallbackWrapper<SkylightObject>(this, &SkylightObject::BottomColorChanged));
}

void
SkylightObject::updatePropsFromSceneLight()
{
  if (!m_sceneLight || !m_sceneLight->m_light)
    return;

  Light* light = m_sceneLight->m_light;

  m_skylightDataObject.TopIntensity.SetValue(light->m_ColorTopIntensity);
  m_skylightDataObject.TopColor.SetValue(glm::vec4(light->m_ColorTop, 1.0f));
  m_skylightDataObject.MiddleIntensity.SetValue(light->m_ColorMiddleIntensity);
  m_skylightDataObject.MiddleColor.SetValue(glm::vec4(light->m_ColorMiddle, 1.0f));
  m_skylightDataObject.BottomIntensity.SetValue(light->m_ColorBottomIntensity);
  m_skylightDataObject.BottomColor.SetValue(glm::vec4(light->m_ColorBottom, 1.0f));
}

void
SkylightObject::updateSceneLightFromProps()
{
  if (!m_sceneLight || !m_sceneLight->m_light)
    return;

  Light* light = m_sceneLight->m_light;

  light->m_ColorTopIntensity = m_skylightDataObject.TopIntensity.GetValue();
  glm::vec4 topColor = m_skylightDataObject.TopColor.GetValue();
  light->m_ColorTop = glm::vec3(topColor.x, topColor.y, topColor.z);

  light->m_ColorMiddleIntensity = m_skylightDataObject.MiddleIntensity.GetValue();
  glm::vec4 middleColor = m_skylightDataObject.MiddleColor.GetValue();
  light->m_ColorMiddle = glm::vec3(middleColor.x, middleColor.y, middleColor.z);

  light->m_ColorBottomIntensity = m_skylightDataObject.BottomIntensity.GetValue();
  glm::vec4 bottomColor = m_skylightDataObject.BottomColor.GetValue();
  light->m_ColorBottom = glm::vec3(bottomColor.x, bottomColor.y, bottomColor.z);

  for (auto& observer : m_sceneLight->m_observers) {
    observer(*light);
  }

  if (m_dirtyCallback) {
    m_dirtyCallback();
  }
}

void
SkylightObject::TopIntensityChanged(prtyProperty* i_Property, bool i_bDirty)
{
  updateSceneLightFromProps();
}

void
SkylightObject::TopColorChanged(prtyProperty* i_Property, bool i_bDirty)
{
  updateSceneLightFromProps();
}

void
SkylightObject::MiddleIntensityChanged(prtyProperty* i_Property, bool i_bDirty)
{
  updateSceneLightFromProps();
}

void
SkylightObject::MiddleColorChanged(prtyProperty* i_Property, bool i_bDirty)
{
  updateSceneLightFromProps();
}

void
SkylightObject::BottomIntensityChanged(prtyProperty* i_Property, bool i_bDirty)
{
  updateSceneLightFromProps();
}

void
SkylightObject::BottomColorChanged(prtyProperty* i_Property, bool i_bDirty)
{
  updateSceneLightFromProps();
}
