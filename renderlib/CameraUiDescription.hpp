#pragma once

#include "CameraDataObject.hpp"
#include "uiInfo.hpp"
#include "core/prty/prtyObject.hpp"

struct CameraUiDescription
{
  static FloatSliderSpinnerUiInfo m_exposure;
  static ComboBoxUiInfo m_exposureIterations;
  static CheckBoxUiInfo m_noiseReduction;
  static FloatSliderSpinnerUiInfo m_apertureSize;
  static FloatSliderSpinnerUiInfo m_fieldOfView;
  static FloatSliderSpinnerUiInfo m_focalDistance;
};

class CameraObject : public prtyObject
{
public:
  CameraObject();

  void updatePropsFromObject();
  void updateObjectFromProps();

private:
  // the properties
  CameraDataObject m_cameraDataObject;

  // the actual camera
  std::shared_ptr<CCamera> m_Camera;

  // the ui info
  FloatSliderSpinnerUiInfo* m_ExposureUIInfo;
  ComboBoxUiInfo* m_ExposureIterationsUIInfo;
  CheckBoxUiInfo* m_NoiseReductionUIInfo;
  FloatSliderSpinnerUiInfo* m_ApertureSizeUIInfo;
  FloatSliderSpinnerUiInfo* m_FieldOfViewUIInfo;
  FloatSliderSpinnerUiInfo* m_FocalDistanceUIInfo;
};
