#pragma once

#include "Camera.h"
#include "Controls.h"

#include <QCheckBox>
#include <QComboBox>
#include <QFormLayout>
#include <QWidget>

class RenderSettings;

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
};
