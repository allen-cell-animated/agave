#include "CameraObject.hpp"

#include "Logging.h"
#include "serialize/docReader.h"
#include "serialize/docWriter.h"
#include "glm.h"

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
  m_ExposureUIInfo->SetToolTip("Set Exposure");
  m_ExposureUIInfo->SetStatusTip("Set camera exposure");
  m_ExposureUIInfo->min = 0.0f;
  m_ExposureUIInfo->max = 1.0f;
  m_ExposureUIInfo->decimals = 2;       // decimals
  m_ExposureUIInfo->singleStep = 0.01f; // singleStep
  m_ExposureUIInfo->numTickMarks = 0;   // numTickMarks
  AddProperty(m_ExposureUIInfo);
  m_ExposureIterationsUIInfo =
    new ComboBoxUiInfo(&m_cameraDataObject.ExposureIterations, "Camera", "Exposure Iterations");
  m_ExposureIterationsUIInfo->SetToolTip("Set Exposure Iterations");
  m_ExposureIterationsUIInfo->SetStatusTip("Set number of samples to accumulate per viewport update");
  AddProperty(m_ExposureIterationsUIInfo);
  m_NoiseReductionUIInfo = new CheckBoxUiInfo(&m_cameraDataObject.NoiseReduction, "Camera", "Noise Reduction");
  m_NoiseReductionUIInfo->SetToolTip("Enable Noise Reduction");
  m_NoiseReductionUIInfo->SetStatusTip("Enable denoising pass");
  AddProperty(m_NoiseReductionUIInfo);
  m_ApertureSizeUIInfo = new FloatSliderSpinnerUiInfo(&m_cameraDataObject.ApertureSize, "Camera", "Aperture Size");
  m_ApertureSizeUIInfo->SetToolTip("Set Aperture Size");
  m_ApertureSizeUIInfo->SetStatusTip("Set camera aperture size");
  m_ApertureSizeUIInfo->min = 0.0f;
  m_ApertureSizeUIInfo->max = 0.1f;
  m_ApertureSizeUIInfo->decimals = 2;       // decimals
  m_ApertureSizeUIInfo->singleStep = 0.01f; //
  m_ApertureSizeUIInfo->numTickMarks = 0;   // numTickMarks
  m_ApertureSizeUIInfo->suffix = " mm";     // suffix
  AddProperty(m_ApertureSizeUIInfo);
  m_FieldOfViewUIInfo = new FloatSliderSpinnerUiInfo(&m_cameraDataObject.FieldOfView, "Camera", "Field of View");
  m_FieldOfViewUIInfo->SetToolTip("Set Field of View");
  m_FieldOfViewUIInfo->SetStatusTip("Set camera field of view angle");
  m_FieldOfViewUIInfo->min = 10.0f;
  m_FieldOfViewUIInfo->max = 150.0f;
  m_FieldOfViewUIInfo->decimals = 2;       // decimals
  m_FieldOfViewUIInfo->singleStep = 0.01f; // single
  m_FieldOfViewUIInfo->numTickMarks = 0;   // numTickMarks
  m_FieldOfViewUIInfo->suffix = " deg";    // suffix
  AddProperty(m_FieldOfViewUIInfo);
  m_FocalDistanceUIInfo = new FloatSliderSpinnerUiInfo(&m_cameraDataObject.FocalDistance, "Camera", "Focal Distance");
  m_FocalDistanceUIInfo->SetToolTip("Set Focal Distance");
  m_FocalDistanceUIInfo->SetStatusTip("Set focal distance");
  m_FocalDistanceUIInfo->min = 0.0f;
  m_FocalDistanceUIInfo->max = 15.0f;
  m_FocalDistanceUIInfo->decimals = 2;       // decimals
  m_FocalDistanceUIInfo->singleStep = 0.01f; // single
  m_FocalDistanceUIInfo->numTickMarks = 0;   // numTickMarks
  m_FocalDistanceUIInfo->suffix = " m";      // suffix
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

  m_cameraDataObject.Position.AddCallback(
    new prtyCallbackWrapper<CameraObject>(this, &CameraObject::TransformationChanged));
  m_cameraDataObject.Target.AddCallback(
    new prtyCallbackWrapper<CameraObject>(this, &CameraObject::TransformationChanged));
  m_cameraDataObject.Roll.AddCallback(
    new prtyCallbackWrapper<CameraObject>(this, &CameraObject::TransformationChanged));
}

void
CameraObject::updatePropsFromObject()
{
  // TODO FIXME if we set everything through props, then this is not needed.
  if (m_camera) {
    m_cameraDataObject.Exposure.SetValue(1.0f - m_camera->m_Film.m_Exposure);

    uint8_t exposureIterationsValue = m_camera->m_Film.m_ExposureIterations;
    // convert m_camera->m_Film.m_ExposureIterations to string
    // and then find the corresponding index in the enum
    switch (m_camera->m_Film.m_ExposureIterations) {
      case 1:
        exposureIterationsValue = 0;
        break;
      case 2:
        exposureIterationsValue = 1;
        break;
      case 4:
        exposureIterationsValue = 2;
        break;
      case 8:
        exposureIterationsValue = 3;
        break;
      default:
        LOG_ERROR << "Invalid Exposure Iterations: " << m_camera->m_Film.m_ExposureIterations;
        exposureIterationsValue = 0; // default to 1
    }
    m_cameraDataObject.ExposureIterations.SetValue(exposureIterationsValue);
    // TODO this is not hooked up to the camera properly
    // m_cameraDataObject.NoiseReduction.SetValue(m_camera->m_Film.m_NoiseReduction);
    m_cameraDataObject.ApertureSize.SetValue(m_camera->m_Aperture.m_Size);
    m_cameraDataObject.FieldOfView.SetValue(m_camera->m_FovV);
    m_cameraDataObject.FocalDistance.SetValue(m_camera->m_Focus.m_FocalDistance);
  }
}
uint8_t
CameraObject::GetExposureIterationsValue(int i_ComboBoxIndex)
{
  // Convert the combo box index to the corresponding exposure iterations value
  switch (i_ComboBoxIndex) {
    case 0:
      return 1; // 1 iteration
    case 1:
      return 2; // 2 iterations
    case 2:
      return 4; // 4 iterations
    case 3:
      return 8; // 8 iterations
    default:
      LOG_ERROR << "Invalid Exposure Iterations index: " << i_ComboBoxIndex;
      return 1; // default to 1 iteration
  }
}

void
CameraObject::updateObjectFromProps()
{
  // update low-level camera object from properties
  if (m_camera) {
    m_camera->m_Film.m_Exposure = 1.0f - m_cameraDataObject.Exposure.GetValue();
    uint8_t exposureIterationsValue = GetExposureIterationsValue(m_cameraDataObject.ExposureIterations.GetValue());
    m_camera->m_Film.m_ExposureIterations = exposureIterationsValue;

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
  m_camera->m_Dirty = true;
}
void
CameraObject::ExposureIterationsChanged(prtyProperty* i_Property, bool i_bDirty)
{
  m_camera->m_Film.m_ExposureIterations = GetExposureIterationsValue(m_cameraDataObject.ExposureIterations.GetValue());
  m_camera->m_Dirty = true;
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
  m_camera->m_Dirty = true;
}
void
CameraObject::FieldOfViewChanged(prtyProperty* i_Property, bool i_bDirty)
{
  m_camera->m_FovV = m_cameraDataObject.FieldOfView.GetValue();
  m_camera->m_Dirty = true;
}
void
CameraObject::FocalDistanceChanged(prtyProperty* i_Property, bool i_bDirty)
{
  m_camera->m_Focus.m_FocalDistance = m_cameraDataObject.FocalDistance.GetValue();
  m_camera->m_Dirty = true;
}

//--------------------------------------------------------------------
// common code when a property related to position of light is changed
//--------------------------------------------------------------------
void
CameraObject::TransformationChanged(prtyProperty* i_Property, bool i_bDirty)
{
  // Rotate up vector through tilt angle
  glm::vec3 pos, target;
  // assumes world space.
  pos = m_cameraDataObject.Position.GetValue();
  target = m_cameraDataObject.Target.GetValue();
  glm::vec3 up = glm::vec3(0, 1, 0); // default up vector
  // Rotate the up vector around the vector from position to target
  // using the roll angle (tilt angle)
  up = glm::rotate(up, DEG_TO_RAD * m_cameraDataObject.Roll.GetValue(), target - pos);

  m_camera->m_From = pos;
  m_camera->m_Target = target;
  m_camera->m_Up = up;
  m_camera->m_Dirty = true;
}

void
CameraObject::fromDocument(docReader* reader)
{
  reader->beginObject("CameraObject");
  reader->readProperties(this);
  reader->endObject();
}
void
CameraObject::toDocument(docWriter* writer)
{
  writer->beginObject("CameraObject");
  // write version property explicitly?
  // ensure that this and most other objects have a (unique) name property?
  writer->writeProperties(this);
  writer->endObject();
}