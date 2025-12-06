#pragma once

#include "uiInfo.hpp"

struct CameraUiDescription
{
  static FloatSliderSpinnerUiInfo m_exposure;
  static ComboBoxUiInfo m_exposureIterations;
  static CheckBoxUiInfo m_noiseReduction;
  static FloatSliderSpinnerUiInfo m_apertureSize;
  static FloatSliderSpinnerUiInfo m_fieldOfView;
  static FloatSliderSpinnerUiInfo m_focalDistance;
};