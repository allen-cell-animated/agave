#include "CameraDataObject.hpp"

#include "Logging.h"

CameraDataObject::CameraDataObject(CCamera* camera)
  : m_camera(camera)
{
  updatePropsFromCamera();
  // hook up properties to update the underlying camera
  Exposure.addCallback([this](prtyProperty<float>* p, bool) { update(); });
  ExposureIterations.addCallback([this](prtyProperty<int>* p, bool) { update(); });
  NoiseReduction.addCallback([this](prtyProperty<bool>* p, bool) { update(); });
  ApertureSize.addCallback([this](prtyProperty<float>* p, bool) { update(); });
  FieldOfView.addCallback([this](prtyProperty<float>* p, bool) { update(); });
  FocalDistance.addCallback([this](prtyProperty<float>* p, bool) { update(); });
}

void
CameraDataObject::updatePropsFromCamera()
{
  if (m_camera) {
    Exposure.set(1.0f - m_camera->m_Film.m_Exposure);
    ExposureIterations.set(m_camera->m_Film.m_ExposureIterations);
    // TODO this is not hooked up to the camera properly
    // NoiseReduction.set(m_camera->m_Film.m_NoiseReduction);
    ApertureSize.set(m_camera->m_Aperture.m_Size);
    FieldOfView.set(m_camera->m_FovV);
    FocalDistance.set(m_camera->m_Focus.m_FocalDistance);
  }
}
void
CameraDataObject::update()
{
  // update low-level camera object from properties
  if (m_camera) {
    m_camera->m_Film.m_Exposure = 1.0f - Exposure.get();
    m_camera->m_Film.m_ExposureIterations = ExposureIterations.get();

    // Aperture
    m_camera->m_Aperture.m_Size = ApertureSize.get();

    // Projection
    m_camera->m_FovV = FieldOfView.get();

    // Focus
    m_camera->m_Focus.m_FocalDistance = FocalDistance.get();

    m_camera->Update();

    // TODO noise reduction!!!

    // TODO how can I hook this up automatically to the RenderSettings dirty flags?
    // renderer should pick this up and do the right thing (TM)
    m_camera->m_Dirty = true;
  }
}