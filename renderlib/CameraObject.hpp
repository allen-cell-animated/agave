#pragma once

#include "CameraDataObject.hpp"
#include "uiInfo.hpp"
#include "core/prty/prtyObject.hpp"
#include "CCamera.h"

class docReader;
class docWriter;

struct CameraUiDescription
{
  static FloatSliderSpinnerUiInfo m_exposure;
  static ComboBoxUiInfo m_exposureIterations;
  static CheckBoxUiInfo m_noiseReduction;
  static FloatSliderSpinnerUiInfo m_apertureSize;
  static FloatSliderSpinnerUiInfo m_fieldOfView;
  static FloatSliderSpinnerUiInfo m_focalDistance;
};

// class IDocumentObject
// {
// public:
//   virtual void fromDocument(docReader* reader) = 0;
//   virtual void toDocument(docWriter* writer) = 0;

//   // necessary for doc reading and writing?
//   prtyName m_Name;
// }

class CameraObject : public prtyObject
{
public:
  CameraObject();

  void updatePropsFromObject();
  void updateObjectFromProps();

  // Getter for camera data object
  CameraDataObject& getCameraDataObject() { return m_cameraDataObject; }
  const CameraDataObject& getCameraDataObject() const { return m_cameraDataObject; }

  // Getters for UI info objects
  FloatSliderSpinnerUiInfo* getExposureUIInfo() { return m_ExposureUIInfo; }
  ComboBoxUiInfo* getExposureIterationsUIInfo() { return m_ExposureIterationsUIInfo; }
  CheckBoxUiInfo* getNoiseReductionUIInfo() { return m_NoiseReductionUIInfo; }
  FloatSliderSpinnerUiInfo* getApertureSizeUIInfo() { return m_ApertureSizeUIInfo; }
  FloatSliderSpinnerUiInfo* getFieldOfViewUIInfo() { return m_FieldOfViewUIInfo; }
  FloatSliderSpinnerUiInfo* getFocalDistanceUIInfo() { return m_FocalDistanceUIInfo; }

  // Getter for the camera
  std::shared_ptr<CCamera> getCamera() const { return m_camera; }

  // Convert UI specific combo box index to a known enum type
  static uint8_t GetExposureIterationsValue(int i_ComboBoxIndex);

  // document reading and writing; TODO consider an abstract base class to enforce commonality
  static constexpr uint32_t CURRENT_VERSION = 1;
  void fromDocument(docReader* reader);
  void toDocument(docWriter* writer);

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
  void TransformationChanged(prtyProperty* i_Property, bool i_bDirty);
};
