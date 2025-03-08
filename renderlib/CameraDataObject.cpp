#include "CameraDataObject.hpp"

#include "Logging.h"

CameraDataObject::CameraDataObject(CCamera* camera)
  : m_camera(camera)
{
  updatePropsFromCamera();
  // hook up properties to update the underlying camera
  Exposure.addCallback([this](prtyProperty<float>* p, bool) {
    // LOG_DEBUG << "Setting exposure to " << p->get();
    update();
  });
  ExposureIterations.addCallback([this](prtyProperty<int>* p, bool) {
    // LOG_DEBUG << "Setting exposure iterations to " << p->get();
    update();
  });
  NoiseReduction.addCallback([this](prtyProperty<bool>* p, bool) {
    // LOG_DEBUG << "Setting noise reduction to " << p->get();
    update();
  });
  ApertureSize.addCallback([this](prtyProperty<float>* p, bool) {
    // LOG_DEBUG << "Setting aperture size to " << p->get();
    update();
  });
  FieldOfView.addCallback([this](prtyProperty<float>* p, bool) {
    // LOG_DEBUG << "Setting field of view to " << p->get();
    update();
  });
  FocalDistance.addCallback([this](prtyProperty<float>* p, bool) {
    // LOG_DEBUG << "Setting focal distance to " << p->get();
    update();
  });
}

void
CameraDataObject::updatePropsFromCamera()
{
  if (m_camera) {
    Exposure.set(1.0f - m_camera->m_Film.m_Exposure);
    ExposureIterations.set(m_camera->m_Film.m_ExposureIterations);
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

    // TODO how can I hook this up automatically to the RenderSettings dirty flags?
    // renderer should pick this up and do the right thing (TM)
    m_camera->m_Dirty = true;
  }
}