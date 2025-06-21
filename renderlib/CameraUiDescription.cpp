#include "CameraUiDescription.hpp"

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
  m_Camera = std::make_shared<CCamera>();
  m_ExposureUIInfo = new FloatSliderSpinnerUiInfo(&m_cameraDataObject.Exposure, "Camera", "Exposure");
  m_ExposureIterationsUIInfo =
    new ComboBoxUiInfo(&m_cameraDataObject.ExposureIterations, "Camera", "Exposure Iterations");
  m_NoiseReductionUIInfo = new CheckBoxUiInfo(&m_cameraDataObject.NoiseReduction, "Camera", "Noise Reduction");
  m_ApertureSizeUIInfo = new FloatSliderSpinnerUiInfo(&m_cameraDataObject.ApertureSize, "Camera", "Aperture Size");
  m_FieldOfViewUIInfo = new FloatSliderSpinnerUiInfo(&m_cameraDataObject.FieldOfView, "Camera", "Field of View");
  m_FocalDistanceUIInfo = new FloatSliderSpinnerUiInfo(&m_cameraDataObject.FocalDistance, "Camera", "Focal Distance");
}
