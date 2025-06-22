#include "CameraUiDescription.hpp"

#include "Logging.h"

// FloatSliderSpinnerUiInfo CameraUiDescription::m_exposure("Exposure",
//                                                          "Set Exposure",
//                                                          "Set camera exposure",
//                                                          0.0f,
//                                                          1.0f,
//                                                          2,    // decimals
//                                                          0.01, // singleStep
//                                                          0     // numTickMarks
// );
// ComboBoxUiInfo CameraUiDescription::m_exposureIterations("Exposure Time",
//                                                          "Set Exposure Time",
//                                                          "Set number of samples to accumulate per viewport update",
//                                                          { "1", "2", "4", "8" });
// CheckBoxUiInfo CameraUiDescription::m_noiseReduction("Noise Reduction",
//                                                      "Enable denoising pass",
//                                                      "Enable denoising pass");
// FloatSliderSpinnerUiInfo CameraUiDescription::m_apertureSize("Aperture Size",
//                                                              "Set camera aperture size",
//                                                              "Set camera aperture size",
//                                                              0.0f,
//                                                              0.1f,
//                                                              2,    // decimals
//                                                              0.01, // singleStep
//                                                              0,    // numTickMarks
//                                                              " mm");
// FloatSliderSpinnerUiInfo CameraUiDescription::m_fieldOfView("Field of view",
//                                                             "Set camera field of view angle",
//                                                             "Set camera field of view angle",
//                                                             10.0f,
//                                                             150.0f,
//                                                             2,    // decimals
//                                                             0.01, // singleStep
//                                                             0,    // numTickMarks
//                                                             " deg");
// FloatSliderSpinnerUiInfo CameraUiDescription::m_focalDistance("Focal distance",
//                                                               "Set focal distance",
//                                                               "Set focal distance",
//                                                               0.0f,
//                                                               15.0f,
//                                                               2,    // decimals
//                                                               0.01, // singleStep
//                                                               0,    // numTickMarks
//                                                               " m");

CameraObject::CameraObject()
  : prtyObject()
{
  m_camera = std::make_shared<CCamera>();
  m_ExposureUIInfo = new FloatSliderSpinnerUiInfo(&m_cameraDataObject.Exposure, "Camera", "Exposure");
  AddProperty(m_ExposureUIInfo);
  m_ExposureIterationsUIInfo =
    new ComboBoxUiInfo(&m_cameraDataObject.ExposureIterations, "Camera", "Exposure Iterations");
  AddProperty(m_ExposureIterationsUIInfo);
  m_NoiseReductionUIInfo = new CheckBoxUiInfo(&m_cameraDataObject.NoiseReduction, "Camera", "Noise Reduction");
  AddProperty(m_NoiseReductionUIInfo);
  m_ApertureSizeUIInfo = new FloatSliderSpinnerUiInfo(&m_cameraDataObject.ApertureSize, "Camera", "Aperture Size");
  AddProperty(m_ApertureSizeUIInfo);
  m_FieldOfViewUIInfo = new FloatSliderSpinnerUiInfo(&m_cameraDataObject.FieldOfView, "Camera", "Field of View");
  AddProperty(m_FieldOfViewUIInfo);
  m_FocalDistanceUIInfo = new FloatSliderSpinnerUiInfo(&m_cameraDataObject.FocalDistance, "Camera", "Focal Distance");
  AddProperty(m_FocalDistanceUIInfo);

  m_cameraDataObject.Exposure.AddCallback(new prtyCallbackWrapper<CameraObject>(this, &CameraObject::ExposureChanged));
  m_cameraDataObject.ExposureIterations.AddCallback(
    new prtyCallbackWrapper<CameraObject>(this, &CameraObject::ExposureIterationsChanged));
  m_cameraDataObject.NoiseReduction.AddCallback(
    new prtyCallbackWrapper<CameraObject>(this, &CameraObject::NoiseReductionChanged));
  m_cameraDataObject.ApertureSize.AddCallback(
    new prtyCallbackWrapper<CameraObject>(this, &CameraObject::ApertureSizeChanged));
  m_cameraDataObject.FieldOfView.AddCallback(
    new prtyCallbackWrapper<CameraObject>(this, &CameraObject::FieldOfViewChanged));
  m_cameraDataObject.FocalDistance.AddCallback(
    new prtyCallbackWrapper<CameraObject>(this, &CameraObject::FocalDistanceChanged));
}

void
CameraObject::updatePropsFromObject()
{
  if (m_camera) {
    m_cameraDataObject.Exposure.SetValue(1.0f - m_camera->m_Film.m_Exposure);
    m_cameraDataObject.ExposureIterations.SetValue(m_camera->m_Film.m_ExposureIterations);
    // TODO this is not hooked up to the camera properly
    // m_cameraDataObject.NoiseReduction.SetValue(m_camera->m_Film.m_NoiseReduction);
    m_cameraDataObject.ApertureSize.SetValue(m_camera->m_Aperture.m_Size);
    m_cameraDataObject.FieldOfView.SetValue(m_camera->m_FovV);
    m_cameraDataObject.FocalDistance.SetValue(m_camera->m_Focus.m_FocalDistance);
  }
}
void
CameraObject::updateObjectFromProps()
{
  // update low-level camera object from properties
  if (m_camera) {
    m_camera->m_Film.m_Exposure = 1.0f - m_cameraDataObject.Exposure.GetValue();
    m_camera->m_Film.m_ExposureIterations = m_cameraDataObject.ExposureIterations.GetValue();

    // Aperture
    m_camera->m_Aperture.m_Size = m_cameraDataObject.ApertureSize.GetValue();

    // Projection
    m_camera->m_FovV = m_cameraDataObject.FieldOfView.GetValue();

    // Focus
    m_camera->m_Focus.m_FocalDistance = m_cameraDataObject.FocalDistance.GetValue();

    m_camera->Update();

    // TODO noise reduction!!!

    // TODO how can I hook this up automatically to the RenderSettings dirty flags?
    // renderer should pick this up and do the right thing (TM)
    m_camera->m_Dirty = true;
  }
}

void
CameraObject::ExposureChanged(prtyProperty* i_Property, bool i_bDirty)
{
  m_camera->m_Film.m_Exposure = 1.0f - m_cameraDataObject.Exposure.GetValue();
}
void
CameraObject::ExposureIterationsChanged(prtyProperty* i_Property, bool i_bDirty)
{
  m_camera->m_Film.m_ExposureIterations = m_cameraDataObject.ExposureIterations.GetValue();
}
void
CameraObject::NoiseReductionChanged(prtyProperty* i_Property, bool i_bDirty)
{
  LOG_ERROR << "Noise reduction is not implemented yet!";
}
void
CameraObject::ApertureSizeChanged(prtyProperty* i_Property, bool i_bDirty)
{
  m_camera->m_Aperture.m_Size = m_cameraDataObject.ApertureSize.GetValue();
}
void
CameraObject::FieldOfViewChanged(prtyProperty* i_Property, bool i_bDirty)
{
  m_camera->m_FovV = m_cameraDataObject.FieldOfView.GetValue();
}
void
CameraObject::FocalDistanceChanged(prtyProperty* i_Property, bool i_bDirty)
{
  m_camera->m_Focus.m_FocalDistance = m_cameraDataObject.FocalDistance.GetValue();
}
