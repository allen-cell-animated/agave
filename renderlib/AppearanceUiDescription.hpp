#pragma once

#include "AppearanceDataObject.hpp"
#include "core/prty/prtyObject.hpp"
#include "RenderSettings.h"
#include "AppScene.h"
#include "uiInfo.hpp"

struct AppearanceUiDescription
{
  static ComboBoxUiInfo m_rendererType;
  static ComboBoxUiInfo m_shadingType;
  static FloatSliderSpinnerUiInfo m_densityScale;
  static FloatSliderSpinnerUiInfo m_gradientFactor;
  static FloatSliderSpinnerUiInfo m_stepSizePrimaryRay;
  static FloatSliderSpinnerUiInfo m_stepSizeSecondaryRay;
  static CheckBoxUiInfo m_interpolate;
  static ColorPickerUiInfo m_backgroundColor;
  static CheckBoxUiInfo m_showBoundingBox;
  static ColorPickerUiInfo m_boundingBoxColor;
  static CheckBoxUiInfo m_showScaleBar;
};

class AppearanceObject : public prtyObject
{
public:
  AppearanceObject();

  void updatePropsFromObject();
  void updateObjectFromProps();

private:
  // the properties
  AppearanceDataObject m_appearanceDataObject;

  // the actual settings
  std::shared_ptr<RenderSettings> m_renderSettings;
  std::weak_ptr<Scene> m_scene;

  // the ui info
  ComboBoxUiInfo* m_rendererType;
  ComboBoxUiInfo* m_shadingType;
  FloatSliderSpinnerUiInfo* m_densityScale;
  FloatSliderSpinnerUiInfo* m_gradientFactor;
  FloatSliderSpinnerUiInfo* m_stepSizePrimaryRay;
  FloatSliderSpinnerUiInfo* m_stepSizeSecondaryRay;
  CheckBoxUiInfo* m_interpolate;
  ColorPickerUiInfo* m_backgroundColor;
  CheckBoxUiInfo* m_showBoundingBox;
  ColorPickerUiInfo* m_boundingBoxColor;
  CheckBoxUiInfo* m_showScaleBar;

  void RendererTypeChanged(prtyProperty* i_Property, bool i_bDirty);
  void ShadingTypeChanged(prtyProperty* i_Property, bool i_bDirty);
  void DensityScaleChanged(prtyProperty* i_Property, bool i_bDirty);
  void GradientFactorChanged(prtyProperty* i_Property, bool i_bDirty);
  void StepSizePrimaryRayChanged(prtyProperty* i_Property, bool i_bDirty);
  void StepSizeSecondaryRayChanged(prtyProperty* i_Property, bool i_bDirty);
  void InterpolateChanged(prtyProperty* i_Property, bool i_bDirty);
  void BackgroundColorChanged(prtyProperty* i_Property, bool i_bDirty);
  void ShowBoundingBoxChanged(prtyProperty* i_Property, bool i_bDirty);
  void BoundingBoxColorChanged(prtyProperty* i_Property, bool i_bDirty);
  void ShowScaleBarChanged(prtyProperty* i_Property, bool i_bDirty);
};
