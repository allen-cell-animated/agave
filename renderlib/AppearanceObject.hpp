#pragma once

#include "AppearanceDataObject.hpp"
#include "core/prty/prtyObject.hpp"
#include "RenderSettings.h"
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

  // the actual camera
  std::shared_ptr<RenderSettings> m_renderSettings;

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
};
