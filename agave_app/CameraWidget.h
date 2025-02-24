#pragma once

#include "Camera.h"
#include "Controls.h"

#include "renderlib/core/prty/prtyProperty.h"
#include "renderlib/Logging.h"

#include <QCheckBox>
#include <QComboBox>
#include <QFormLayout>
#include <QWidget>

class RenderSettings;

class CameraDataObject
{
public:
  CameraDataObject()
    : m_camera(nullptr)
  {
    // hook up properties to update the underlying camera
    Exposure.addCallback([this](prtyProperty<float>* p, bool) {
      LOG_DEBUG << "Setting exposure to " << p->get();
      // m_camera->GetFilm().SetExposure(val);
    });
    ExposureIterations.addCallback([this](prtyProperty<int>* p, bool) {
      LOG_DEBUG << "Setting exposure iterations to " << p->get();
      // m_camera->GetFilm().SetExposureIterations(val);
    });
    NoiseReduction.addCallback([this](prtyProperty<bool>* p, bool) {
      LOG_DEBUG << "Setting noise reduction to " << p->get();
      // m_camera->GetFilm().SetNoiseReduction(val);
    });
    ApertureSize.addCallback([this](prtyProperty<float>* p, bool) {
      LOG_DEBUG << "Setting aperture size to " << p->get();
      // m_camera->GetAperture().SetSize(val);
    });
    FieldOfView.addCallback([this](prtyProperty<float>* p, bool) {
      LOG_DEBUG << "Setting field of view to " << p->get();
      // m_camera->GetProjection().SetFieldOfView(val);
    });
    FocalDistance.addCallback([this](prtyProperty<float>* p, bool) {
      LOG_DEBUG << "Setting focal distance to " << p->get();
      // m_camera->GetFocus().SetFocalDistance(val);
    });
  }
  CameraDataObject(CCamera* camera)
    : m_camera(camera)
  {
    // hook up properties to update the underlying camera
    Exposure.addCallback([this](prtyProperty<float>* p, bool) {
      LOG_DEBUG << "Setting exposure to " << p->get();
      // m_camera->GetFilm().SetExposure(val);
    });
    ExposureIterations.addCallback([this](prtyProperty<int>* p, bool) {
      LOG_DEBUG << "Setting exposure iterations to " << p->get();
      // m_camera->GetFilm().SetExposureIterations(val);
    });
    NoiseReduction.addCallback([this](prtyProperty<bool>* p, bool) {
      LOG_DEBUG << "Setting noise reduction to " << p->get();
      // m_camera->GetFilm().SetNoiseReduction(val);
    });
    ApertureSize.addCallback([this](prtyProperty<float>* p, bool) {
      LOG_DEBUG << "Setting aperture size to " << p->get();
      // m_camera->GetAperture().SetSize(val);
    });
    FieldOfView.addCallback([this](prtyProperty<float>* p, bool) {
      LOG_DEBUG << "Setting field of view to " << p->get();
      // m_camera->GetProjection().SetFieldOfView(val);
    });
    FocalDistance.addCallback([this](prtyProperty<float>* p, bool) {
      LOG_DEBUG << "Setting focal distance to " << p->get();
      // m_camera->GetFocus().SetFocalDistance(val);
    });
  }
  prtyProperty<float> Exposure{ "Exposure", 0.75f };
  prtyProperty<int> ExposureIterations{ "ExposureIterations", 1 };
  prtyProperty<bool> NoiseReduction{ "NoiseReduction", false };
  prtyProperty<float> ApertureSize{ "ApertureSize", 0.0f };
  prtyProperty<float> FieldOfView{ "FieldOfView", 30.0f };
  prtyProperty<float> FocalDistance{ "FocalDistance", 0.0f };

  CCamera* m_camera;
};

class QCameraWidget : public QWidget
{
  Q_OBJECT

public:
  QCameraWidget(QWidget* pParent = NULL, QCamera* cam = nullptr, RenderSettings* rs = nullptr);

  virtual QSize sizeHint() const;

private:
  QFormLayout m_MainLayout;

  QCamera* m_qcamera;
  RenderSettings* m_renderSettings;

  QNumericSlider m_ExposureSlider;
  QComboBox m_ExposureIterationsSpinner;
  QCheckBox m_NoiseReduction;
  QNumericSlider m_ApertureSizeSlider;
  QNumericSlider m_FieldOfViewSlider;
  QNumericSlider m_FocalDistanceSlider;

  void SetExposure(const double& Exposure);
  void SetExposureIterations(int index);
  void OnNoiseReduction(const int& ReduceNoise);
  void SetAperture(const double& Aperture);
  void SetFieldOfView(const double& FieldOfView);
  void SetFocalDistance(const double& FocalDistance);

private slots:
  void OnFilmChanged();
  void OnApertureChanged();
  void OnFocusChanged();
  void OnProjectionChanged();

private:
  CameraDataObject m_cameraProps;
};
