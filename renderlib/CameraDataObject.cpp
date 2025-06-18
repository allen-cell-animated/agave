#include "CameraDataObject.hpp"

#include "Logging.h"

CameraDataObject::CameraDataObject(CCamera* camera)
  : m_camera(camera)
{
  updatePropsFromCamera();
  // hook up properties to update the underlying camera
  // Exposure.addCallback([this](prtyProperty<float>* p, bool fromUi) {
  //   if (fromUi)
  //     update();
  // });
  // ExposureIterations.addCallback([this](prtyProperty<int>* p, bool fromUi) {
  //   if (fromUi)
  //     update();
  // });
  // NoiseReduction.addCallback([this](prtyProperty<bool>* p, bool fromUi) {
  //   if (fromUi)
  //     update();
  // });
  // ApertureSize.addCallback([this](prtyProperty<float>* p, bool fromUi) {
  //   if (fromUi)
  //     update();
  // });
  // FieldOfView.addCallback([this](prtyProperty<float>* p, bool fromUi) {
  //   if (fromUi)
  //     update();
  // });
  // FocalDistance.addCallback([this](prtyProperty<float>* p, bool fromUi) {
  //   if (fromUi)
  //     update();
  // });
}

void
CameraDataObject::updatePropsFromCamera()
{
  if (m_camera) {
    Exposure.SetValue(1.0f - m_camera->m_Film.m_Exposure);
    ExposureIterations.SetValue(m_camera->m_Film.m_ExposureIterations);
    // TODO this is not hooked up to the camera properly
    // NoiseReduction.set(m_camera->m_Film.m_NoiseReduction);
    ApertureSize.SetValue(m_camera->m_Aperture.m_Size);
    FieldOfView.SetValue(m_camera->m_FovV);
    FocalDistance.SetValue(m_camera->m_Focus.m_FocalDistance);
  }
}
void
CameraDataObject::update()
{
  // update low-level camera object from properties
  if (m_camera) {
    m_camera->m_Film.m_Exposure = 1.0f - Exposure.GetValue();
    m_camera->m_Film.m_ExposureIterations = ExposureIterations.GetValue();

    // Aperture
    m_camera->m_Aperture.m_Size = ApertureSize.GetValue();

    // Projection
    m_camera->m_FovV = FieldOfView.GetValue();

    // Focus
    m_camera->m_Focus.m_FocalDistance = FocalDistance.GetValue();

    m_camera->Update();

    // renderer should pick this up and do the right thing (TM)
    m_camera->m_Dirty = true;
  }
}