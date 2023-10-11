#include "CameraWidget.h"
#include "RenderSettings.h"

QCameraWidget::QCameraWidget(QWidget* pParent, QCamera* cam, RenderSettings* rs)
  : QWidget(pParent)
  , m_MainLayout()
  , m_qcamera(cam)
  , m_renderSettings(rs)
{
  Controls::initFormLayout(m_MainLayout);
  setLayout(&m_MainLayout);

  // Exposure, controls how bright or dim overall scene is
  m_ExposureSlider.setStatusTip(tr("Set Exposure"));
  m_ExposureSlider.setToolTip(tr("Set camera exposure"));
  m_ExposureSlider.setRange(0.0f, 1.0f);
  m_ExposureSlider.setValue(cam->GetFilm().GetExposure());
  m_ExposureSlider.setDecimals(2);
  m_ExposureSlider.setSingleStep(0.01);

  m_MainLayout.addRow("Exposure", &m_ExposureSlider);

  connect(&m_ExposureSlider, &QNumericSlider::valueChanged, this, &QCameraWidget::SetExposure);

  // Number of render iterations per viewport update
  m_ExposureIterationsSpinner.setStatusTip(tr("Set Exposure Time"));
  m_ExposureIterationsSpinner.setToolTip(tr("Set number of samples to accumulate per viewport update"));
  m_ExposureIterationsSpinner.addItem("1", 1);
  m_ExposureIterationsSpinner.addItem("2", 2);
  m_ExposureIterationsSpinner.addItem("4", 4);
  m_ExposureIterationsSpinner.addItem("8", 8);
  m_ExposureIterationsSpinner.setCurrentIndex(
    m_ExposureIterationsSpinner.findData(cam->GetFilm().GetExposureIterations()));
  m_MainLayout.addRow("Exposure Time", &m_ExposureIterationsSpinner);
  connect(&m_ExposureIterationsSpinner, &QComboBox::currentIndexChanged, this, &QCameraWidget::SetExposureIterations);

  m_NoiseReduction.setStatusTip(tr("Enable denoising pass"));
  m_NoiseReduction.setToolTip(tr("Enable denoising pass"));
  m_NoiseReduction.setCheckState(rs->m_DenoiseParams.m_Enabled ? Qt::CheckState::Checked : Qt::CheckState::Unchecked);
  m_MainLayout.addRow("Noise Reduction", &m_NoiseReduction);

  connect(&m_NoiseReduction, &QCheckBox::stateChanged, this, &QCameraWidget::OnNoiseReduction);

  m_ApertureSizeSlider.setStatusTip(tr("Set camera aperture size"));
  m_ApertureSizeSlider.setToolTip(tr("Set camera aperture size"));
  m_ApertureSizeSlider.setRange(0.0, 0.1);
  m_ApertureSizeSlider.setSuffix(" mm");
  m_ApertureSizeSlider.setDecimals(2);
  m_ApertureSizeSlider.setValue(0.0);
  m_ApertureSizeSlider.setSingleStep(0.01);
  m_MainLayout.addRow("Aperture Size", &m_ApertureSizeSlider);

  connect(&m_ApertureSizeSlider, &QNumericSlider::valueChanged, this, &QCameraWidget::SetAperture);

  m_FieldOfViewSlider.setStatusTip(tr("Set camera field of view angle"));
  m_FieldOfViewSlider.setToolTip(tr("Set camera field of view angle"));
  m_FieldOfViewSlider.setRange(10.0, 150.0);
  m_FieldOfViewSlider.setDecimals(2);
  m_FieldOfViewSlider.setValue(cam->GetProjection().GetFieldOfView());
  m_FieldOfViewSlider.setSuffix(" deg.");
  m_MainLayout.addRow("Field of view", &m_FieldOfViewSlider);

  connect(&m_FieldOfViewSlider, &QNumericSlider::valueChanged, this, &QCameraWidget::SetFieldOfView);

  // Focal distance
  m_FocalDistanceSlider.setStatusTip(tr("Set focal distance"));
  m_FocalDistanceSlider.setToolTip(tr("Set focal distance"));
  m_FocalDistanceSlider.setRange(0.0, 15.0);
  m_FocalDistanceSlider.setDecimals(2);
  m_FocalDistanceSlider.setValue(0.0);
  m_FocalDistanceSlider.setSuffix(" m");

  m_MainLayout.addRow("Focal distance", &m_FocalDistanceSlider);

  connect(&m_FocalDistanceSlider, &QNumericSlider::valueChanged, this, &QCameraWidget::SetFocalDistance);

  QObject::connect(&cam->GetFilm(), SIGNAL(Changed(const QFilm&)), cam, SLOT(OnFilmChanged()));
  QObject::connect(&cam->GetAperture(), SIGNAL(Changed(const QAperture&)), cam, SLOT(OnApertureChanged()));
  QObject::connect(&cam->GetProjection(), SIGNAL(Changed(const QProjection&)), cam, SLOT(OnProjectionChanged()));
  QObject::connect(&cam->GetFocus(), SIGNAL(Changed(const QFocus&)), cam, SLOT(OnFocusChanged()));
}

QSize
QCameraWidget::sizeHint() const
{
  return QSize(20, 20);
}

void
QCameraWidget::SetExposure(const double& Exposure)
{
  m_qcamera->GetFilm().SetExposure(Exposure);
}

void
QCameraWidget::SetExposureIterations(int index)
{
  int value = m_ExposureIterationsSpinner.currentData().toInt();
  m_qcamera->GetFilm().SetExposureIterations(value);
}

void
QCameraWidget::OnNoiseReduction(const int& ReduceNoise)
{
  m_qcamera->GetFilm().SetNoiseReduction(m_NoiseReduction.checkState());
}

void
QCameraWidget::SetAperture(const double& Aperture)
{
  m_qcamera->GetAperture().SetSize(Aperture);
}

void
QCameraWidget::SetFieldOfView(const double& FieldOfView)
{
  m_qcamera->GetProjection().SetFieldOfView(FieldOfView);
}

void
QCameraWidget::SetFocalDistance(const double& FocalDistance)
{
  m_qcamera->GetFocus().SetFocalDistance(FocalDistance);
}
