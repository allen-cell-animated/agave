#pragma once

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
