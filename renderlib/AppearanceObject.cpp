#include "AppearanceObject.hpp"

#include "Logging.h"
// ComboBoxUiInfo AppearanceUiDescription::m_rendererType("Renderer Type",
//                                                        "Select volume rendering type",
//                                                        "Select volume rendering type",
//                                                        { "Ray march blending", "Path Traced" });
// ComboBoxUiInfo AppearanceUiDescription::m_shadingType("Shading Type",
//                                                       "Select volume shading style",
//                                                       "Select volume shading style",
//                                                       { "BRDF Only", "Phase Function Only", "Mixed" });
// FloatSliderSpinnerUiInfo AppearanceUiDescription::m_densityScale("Scattering Density",
//                                                                  "Set scattering density for volume",
//                                                                  "Set scattering density for volume",
//                                                                  0.001f,
//                                                                  100.0f,
//                                                                  3,
//                                                                  0.01f,
//                                                                  10);
// FloatSliderSpinnerUiInfo AppearanceUiDescription::m_gradientFactor("Shading Type Mixture",
//                                                                    "Mix between BRDF and Phase shading",
//                                                                    "Mix between BRDF and Phase shading",
//                                                                    0.0f,
//                                                                    1.0f,
//                                                                    3,
//                                                                    0.01f,
//                                                                    10);
// FloatSliderSpinnerUiInfo AppearanceUiDescription::m_stepSizePrimaryRay("Primary Ray Step Size",
//                                                                        "Set volume ray march step size for camera
//                                                                        rays", "Set volume ray march step size for
//                                                                        camera rays", 1.0f, 100.0f, 3, 0.01f, 10);
// FloatSliderSpinnerUiInfo AppearanceUiDescription::m_stepSizeSecondaryRay(
//   "Secondary Ray Step Size",
//   "Set volume ray march step size for scattered rays",
//   "Set volume ray march step size for scattered rays",
//   1.0f,
//   100.0f,
//   3,
//   0.01f,
//   10);
// CheckBoxUiInfo AppearanceUiDescription::m_interpolate("Interpolate",
//                                                       "Interpolated volume sampling",
//                                                       "Interpolated volume sampling");
// ColorPickerUiInfo AppearanceUiDescription::m_backgroundColor("Background Color",
//                                                              "Set background color",
//                                                              "Set background color");
// CheckBoxUiInfo AppearanceUiDescription::m_showBoundingBox("Show Bounding Box",
//                                                           "Show/hide bounding box",
//                                                           "Show/hide bounding box");
// ColorPickerUiInfo AppearanceUiDescription::m_boundingBoxColor("Bounding Box Color",
//                                                               "Set bounding box color",
//                                                               "Set bounding box color");
// CheckBoxUiInfo AppearanceUiDescription::m_showScaleBar("Show Scale Bar", "Show/hide scale bar", "Show/hide scale
// bar");

AppearanceObject::AppearanceObject()
  : prtyObject()
{
  std::string category("Rendering");
  m_renderSettings = std::make_shared<RenderSettings>();
  m_rendererType = new ComboBoxUiInfo(&m_appearanceDataObject.RendererType, category, "Renderer Type");
  m_rendererType->SetToolTip("Select volume rendering type");
  m_rendererType->SetStatusTip("Select volume rendering type");
  AddProperty(m_rendererType);
  m_shadingType = new ComboBoxUiInfo(&m_appearanceDataObject.ShadingType, category, "Shading Type");
  m_shadingType->SetToolTip("Select volume shading style");
  m_shadingType->SetStatusTip("Select volume shading style");
  AddProperty(m_shadingType);
  m_densityScale = new FloatSliderSpinnerUiInfo(&m_appearanceDataObject.DensityScale, category, "Density Scale");
  m_densityScale->SetToolTip("Set scattering density for volume");
  m_densityScale->SetStatusTip("Set scattering density for volume");
  m_densityScale->min = 0.001f;
  m_densityScale->max = 100.0f;
  m_densityScale->decimals = 3;       // decimals
  m_densityScale->singleStep = 0.01f; // singleStep
  m_densityScale->numTickMarks = 10;  // numTickMarks
  m_densityScale->suffix = "";        // suffix
  AddProperty(m_densityScale);
  m_gradientFactor = new FloatSliderSpinnerUiInfo(&m_appearanceDataObject.GradientFactor, category, "Gradient Factor");
  m_gradientFactor->SetToolTip("Mix between BRDF and Phase shading");
  m_gradientFactor->SetStatusTip("Mix between BRDF and Phase shading");
  m_gradientFactor->min = 0.0f;
  m_gradientFactor->max = 1.0f;
  m_gradientFactor->decimals = 3;       // decimals
  m_gradientFactor->singleStep = 0.01f; // singleStep
  m_gradientFactor->numTickMarks = 10;  // numTickMarks
  m_gradientFactor->suffix = "";        // suffix
  AddProperty(m_gradientFactor);
  m_stepSizePrimaryRay =
    new FloatSliderSpinnerUiInfo(&m_appearanceDataObject.StepSizePrimaryRay, category, "Step Size Primary Ray");
  m_stepSizePrimaryRay->SetToolTip("Set volume ray march step size for camera rays");
  m_stepSizePrimaryRay->SetStatusTip("Set volume ray march step size for camera rays");
  m_stepSizePrimaryRay->min = 1.0f;
  m_stepSizePrimaryRay->max = 100.0f;
  m_stepSizePrimaryRay->decimals = 3;       // decimals
  m_stepSizePrimaryRay->singleStep = 0.01f; // singleStep
  m_stepSizePrimaryRay->numTickMarks = 10;  // numTickMarks
  m_stepSizePrimaryRay->suffix = "";        // suffix
  AddProperty(m_stepSizePrimaryRay);
  m_stepSizeSecondaryRay =
    new FloatSliderSpinnerUiInfo(&m_appearanceDataObject.StepSizeSecondaryRay, category, "Step Size Secondary Ray");
  m_stepSizeSecondaryRay->SetToolTip("Set volume ray march step size for scattered rays");
  m_stepSizeSecondaryRay->SetStatusTip("Set volume ray march step size for scattered rays");
  m_stepSizeSecondaryRay->min = 1.0f;
  m_stepSizeSecondaryRay->max = 100.0f;
  m_stepSizeSecondaryRay->decimals = 3;       // decimals
  m_stepSizeSecondaryRay->singleStep = 0.01f; // singleStep
  m_stepSizeSecondaryRay->numTickMarks = 10;  // numTickMarks
  m_stepSizeSecondaryRay->suffix = "";        // suffix
  AddProperty(m_stepSizeSecondaryRay);
  m_interpolate = new CheckBoxUiInfo(&m_appearanceDataObject.Interpolate, category, "Interpolate");
  m_interpolate->SetToolTip("Interpolated volume sampling");
  m_interpolate->SetStatusTip("Interpolated volume sampling");
  AddProperty(m_interpolate);
  m_backgroundColor = new ColorPickerUiInfo(&m_appearanceDataObject.BackgroundColor, category, "Background Color");
  m_backgroundColor->SetToolTip("Set background color");
  m_backgroundColor->SetStatusTip("Set background color");
  AddProperty(m_backgroundColor);
  m_showBoundingBox = new CheckBoxUiInfo(&m_appearanceDataObject.ShowBoundingBox, category, "Show Bounding Box");
  m_showBoundingBox->SetToolTip("Show/hide bounding box");
  m_showBoundingBox->SetStatusTip("Show/hide bounding box");
  AddProperty(m_showBoundingBox);
  m_boundingBoxColor = new ColorPickerUiInfo(&m_appearanceDataObject.BoundingBoxColor, category, "Bounding Box Color");
  m_boundingBoxColor->SetToolTip("Set bounding box color");
  m_boundingBoxColor->SetStatusTip("Set bounding box color");
  AddProperty(m_boundingBoxColor);
  m_showScaleBar = new CheckBoxUiInfo(&m_appearanceDataObject.ShowScaleBar, category, "Show Scale Bar");
  m_showScaleBar->SetToolTip("Show/hide scale bar");
  m_showScaleBar->SetStatusTip("Show/hide scale bar");
  AddProperty(m_showScaleBar);

  m_appearanceDataObject.RendererType.AddCallback(
    new prtyCallbackWrapper<AppearanceObject>(this, &AppearanceObject::RendererTypeChanged));
  m_appearanceDataObject.ShadingType.AddCallback(
    new prtyCallbackWrapper<AppearanceObject>(this, &AppearanceObject::ShadingTypeChanged));
  m_appearanceDataObject.DensityScale.AddCallback(
    new prtyCallbackWrapper<AppearanceObject>(this, &AppearanceObject::DensityScaleChanged));
  m_appearanceDataObject.GradientFactor.AddCallback(
    new prtyCallbackWrapper<AppearanceObject>(this, &AppearanceObject::GradientFactorChanged));
  m_appearanceDataObject.StepSizePrimaryRay.AddCallback(
    new prtyCallbackWrapper<AppearanceObject>(this, &AppearanceObject::StepSizePrimaryRayChanged));
  m_appearanceDataObject.StepSizeSecondaryRay.AddCallback(
    new prtyCallbackWrapper<AppearanceObject>(this, &AppearanceObject::StepSizeSecondaryRayChanged));
  m_appearanceDataObject.Interpolate.AddCallback(
    new prtyCallbackWrapper<AppearanceObject>(this, &AppearanceObject::InterpolateChanged));
  m_appearanceDataObject.BackgroundColor.AddCallback(
    new prtyCallbackWrapper<AppearanceObject>(this, &AppearanceObject::BackgroundColorChanged));
  m_appearanceDataObject.ShowBoundingBox.AddCallback(
    new prtyCallbackWrapper<AppearanceObject>(this, &AppearanceObject::ShowBoundingBoxChanged));
  m_appearanceDataObject.BoundingBoxColor.AddCallback(
    new prtyCallbackWrapper<AppearanceObject>(this, &AppearanceObject::BoundingBoxColorChanged));
  m_appearanceDataObject.ShowScaleBar.AddCallback(
    new prtyCallbackWrapper<AppearanceObject>(this, &AppearanceObject::ShowScaleBarChanged));
}

void
AppearanceObject::updatePropsFromObject()
{
  if (m_renderSettings) {
    m_appearanceDataObject.ShadingType.SetValue(m_renderSettings->m_RenderSettings.m_ShadingType);
    m_appearanceDataObject.RendererType.SetValue(m_renderSettings->m_rendererType);
    m_appearanceDataObject.DensityScale.SetValue(m_renderSettings->m_RenderSettings.m_DensityScale);
    m_appearanceDataObject.GradientFactor.SetValue(m_renderSettings->m_RenderSettings.m_GradientFactor);
    m_appearanceDataObject.StepSizePrimaryRay.SetValue(m_renderSettings->m_RenderSettings.m_StepSizeFactor);
    m_appearanceDataObject.StepSizeSecondaryRay.SetValue(m_renderSettings->m_RenderSettings.m_StepSizeFactorShadow);
    m_appearanceDataObject.Interpolate.SetValue(m_renderSettings->m_RenderSettings.m_InterpolatedVolumeSampling);
  }
  if (auto scene = m_scene.lock()) {
    m_appearanceDataObject.BackgroundColor.SetValue(glm::vec4(scene->m_material.m_backgroundColor[0],
                                                              scene->m_material.m_backgroundColor[1],
                                                              scene->m_material.m_backgroundColor[2],
                                                              1.0f));
    m_appearanceDataObject.ShowBoundingBox.SetValue(scene->m_material.m_showBoundingBox);
    m_appearanceDataObject.BoundingBoxColor.SetValue(glm::vec4(scene->m_material.m_boundingBoxColor[0],
                                                               scene->m_material.m_boundingBoxColor[1],
                                                               scene->m_material.m_boundingBoxColor[2],
                                                               1.0f));
    m_appearanceDataObject.ShowScaleBar.SetValue(scene->m_showScaleBar);
  }
}

void
AppearanceObject::updateObjectFromProps()
{
  // update low-level object from properties
  if (m_renderSettings) {
    m_renderSettings->m_RenderSettings.m_ShadingType = m_appearanceDataObject.ShadingType.GetValue();
    m_renderSettings->m_rendererType = m_appearanceDataObject.RendererType.GetValue();
    m_renderSettings->m_RenderSettings.m_DensityScale = m_appearanceDataObject.DensityScale.GetValue();
    m_renderSettings->m_RenderSettings.m_GradientFactor = m_appearanceDataObject.GradientFactor.GetValue();
    m_renderSettings->m_RenderSettings.m_StepSizeFactor = m_appearanceDataObject.StepSizePrimaryRay.GetValue();
    m_renderSettings->m_RenderSettings.m_StepSizeFactorShadow = m_appearanceDataObject.StepSizeSecondaryRay.GetValue();
    m_renderSettings->m_RenderSettings.m_InterpolatedVolumeSampling = m_appearanceDataObject.Interpolate.GetValue();
    if (auto scene = m_scene.lock()) {
      scene->m_material.m_backgroundColor[0] = m_appearanceDataObject.BackgroundColor.GetValue().x;
      scene->m_material.m_backgroundColor[1] = m_appearanceDataObject.BackgroundColor.GetValue().y;
      scene->m_material.m_backgroundColor[2] = m_appearanceDataObject.BackgroundColor.GetValue().z;
      scene->m_material.m_showBoundingBox = m_appearanceDataObject.ShowBoundingBox.GetValue();
      scene->m_material.m_boundingBoxColor[0] = m_appearanceDataObject.BoundingBoxColor.GetValue().x;
      scene->m_material.m_boundingBoxColor[1] = m_appearanceDataObject.BoundingBoxColor.GetValue().y;
      scene->m_material.m_boundingBoxColor[2] = m_appearanceDataObject.BoundingBoxColor.GetValue().z;
      scene->m_showScaleBar = m_appearanceDataObject.ShowScaleBar.GetValue();
    }
    m_renderSettings->m_DirtyFlags.SetFlag(RenderParamsDirty);
    m_renderSettings->m_DirtyFlags.SetFlag(TransferFunctionDirty);
    m_renderSettings->m_DirtyFlags.SetFlag(LightsDirty);
  }
}

void
AppearanceObject::RendererTypeChanged(prtyProperty* i_Property, bool i_bDirty)
{
  LOG_ERROR << "Renderer type changed, but not implemented yet!";
}
void
AppearanceObject::ShadingTypeChanged(prtyProperty* i_Property, bool i_bDirty)
{
  m_renderSettings->m_RenderSettings.m_ShadingType = m_appearanceDataObject.ShadingType.GetValue();
  m_renderSettings->m_DirtyFlags.SetFlag(RenderParamsDirty);
}
void
AppearanceObject::DensityScaleChanged(prtyProperty* i_Property, bool i_bDirty)
{
  m_renderSettings->m_RenderSettings.m_DensityScale = m_appearanceDataObject.DensityScale.GetValue();
  m_renderSettings->m_DirtyFlags.SetFlag(RenderParamsDirty);
}
void
AppearanceObject::GradientFactorChanged(prtyProperty* i_Property, bool i_bDirty)
{
  m_renderSettings->m_RenderSettings.m_GradientFactor = m_appearanceDataObject.GradientFactor.GetValue();
  m_renderSettings->m_DirtyFlags.SetFlag(RenderParamsDirty);
}
void
AppearanceObject::StepSizePrimaryRayChanged(prtyProperty* i_Property, bool i_bDirty)
{
  m_renderSettings->m_RenderSettings.m_StepSizeFactor = m_appearanceDataObject.StepSizePrimaryRay.GetValue();
  m_renderSettings->m_DirtyFlags.SetFlag(RenderParamsDirty);
}
void
AppearanceObject::StepSizeSecondaryRayChanged(prtyProperty* i_Property, bool i_bDirty)
{
  m_renderSettings->m_RenderSettings.m_StepSizeFactorShadow = m_appearanceDataObject.StepSizeSecondaryRay.GetValue();
  m_renderSettings->m_DirtyFlags.SetFlag(RenderParamsDirty);
}
void
AppearanceObject::InterpolateChanged(prtyProperty* i_Property, bool i_bDirty)
{
  m_renderSettings->m_RenderSettings.m_InterpolatedVolumeSampling = m_appearanceDataObject.Interpolate.GetValue();
  m_renderSettings->m_DirtyFlags.SetFlag(RenderParamsDirty);
}
void
AppearanceObject::BackgroundColorChanged(prtyProperty* i_Property, bool i_bDirty)
{
  if (auto scene = m_scene.lock()) {
    scene->m_material.m_backgroundColor[0] = m_appearanceDataObject.BackgroundColor.GetValue().x;
    scene->m_material.m_backgroundColor[1] = m_appearanceDataObject.BackgroundColor.GetValue().y;
    scene->m_material.m_backgroundColor[2] = m_appearanceDataObject.BackgroundColor.GetValue().z;
    m_renderSettings->m_DirtyFlags.SetFlag(EnvironmentDirty);
  }
}
void
AppearanceObject::ShowBoundingBoxChanged(prtyProperty* i_Property, bool i_bDirty)
{
  if (auto scene = m_scene.lock()) {
    scene->m_material.m_showBoundingBox = m_appearanceDataObject.ShowBoundingBox.GetValue();
    m_renderSettings->m_DirtyFlags.SetFlag(EnvironmentDirty);
  }
}
void
AppearanceObject::BoundingBoxColorChanged(prtyProperty* i_Property, bool i_bDirty)
{
  if (auto scene = m_scene.lock()) {
    scene->m_material.m_boundingBoxColor[0] = m_appearanceDataObject.BoundingBoxColor.GetValue().x;
    scene->m_material.m_boundingBoxColor[1] = m_appearanceDataObject.BoundingBoxColor.GetValue().y;
    scene->m_material.m_boundingBoxColor[2] = m_appearanceDataObject.BoundingBoxColor.GetValue().z;
    m_renderSettings->m_DirtyFlags.SetFlag(EnvironmentDirty);
  }
}
void
AppearanceObject::ShowScaleBarChanged(prtyProperty* i_Property, bool i_bDirty)
{
  if (auto scene = m_scene.lock()) {
    scene->m_showScaleBar = m_appearanceDataObject.ShowScaleBar.GetValue();
    m_renderSettings->m_DirtyFlags.SetFlag(EnvironmentDirty);
  }
}
