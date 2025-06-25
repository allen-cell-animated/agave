#pragma once

#include "CameraDataObject.hpp"
#include "uiInfo.hpp"
#include "core/prty/prtyObject.hpp"
#include "CCamera.h"

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

  // Getter for camera data object
  // CameraDataObject& getCameraDataObject() { return m_cameraDataObject; }
  const CameraDataObject& getCameraDataObject() const { return m_cameraDataObject; }

  // Getters for UI info objects
  FloatSliderSpinnerUiInfo* getExposureUIInfo() { return m_ExposureUIInfo; }
  ComboBoxUiInfo* getExposureIterationsUIInfo() { return m_ExposureIterationsUIInfo; }
  CheckBoxUiInfo* getNoiseReductionUIInfo() { return m_NoiseReductionUIInfo; }
  FloatSliderSpinnerUiInfo* getApertureSizeUIInfo() { return m_ApertureSizeUIInfo; }
  FloatSliderSpinnerUiInfo* getFieldOfViewUIInfo() { return m_FieldOfViewUIInfo; }
  FloatSliderSpinnerUiInfo* getFocalDistanceUIInfo() { return m_FocalDistanceUIInfo; }

private:
  // the properties
  CameraDataObject m_cameraDataObject;

  // the actual camera
  std::shared_ptr<CCamera> m_camera;

  // the ui info
  FloatSliderSpinnerUiInfo* m_ExposureUIInfo;
  ComboBoxUiInfo* m_ExposureIterationsUIInfo;
  CheckBoxUiInfo* m_NoiseReductionUIInfo;
  FloatSliderSpinnerUiInfo* m_ApertureSizeUIInfo;
  FloatSliderSpinnerUiInfo* m_FieldOfViewUIInfo;
  FloatSliderSpinnerUiInfo* m_FocalDistanceUIInfo;

  void ExposureChanged(prtyProperty* i_Property, bool i_bDirty);
  void ExposureIterationsChanged(prtyProperty* i_Property, bool i_bDirty);
  void NoiseReductionChanged(prtyProperty* i_Property, bool i_bDirty);
  void ApertureSizeChanged(prtyProperty* i_Property, bool i_bDirty);
  void FieldOfViewChanged(prtyProperty* i_Property, bool i_bDirty);
  void FocalDistanceChanged(prtyProperty* i_Property, bool i_bDirty);
};
