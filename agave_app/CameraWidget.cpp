#include "CameraWidget.h"
#include "RenderSettings.h"

#include "renderlib/uiInfo.hpp"

#include <QLabel>
#include <QFormLayout>
#include <map>

QCameraWidget::QCameraWidget(QWidget* pParent, RenderSettings* rs, CameraObject* cameraObject)
  : QWidget(pParent)
  , m_MainLayout()
  , m_renderSettings(rs)
  , m_cameraObject(cameraObject)
{
  Controls::initFormLayout(m_MainLayout);
  setLayout(&m_MainLayout);

  QNumericSlider* slider = addRow(*m_cameraObject->getExposureUIInfo());
  m_MainLayout.addRow("Exposure", slider);
  QComboBox* comboBox = addRow(*m_cameraObject->getExposureIterationsUIInfo());
  m_MainLayout.addRow("Exposure Time", comboBox);
  QCheckBox* checkBox = addRow(*m_cameraObject->getNoiseReductionUIInfo());
  m_MainLayout.addRow("Noise Reduction", checkBox);
  QNumericSlider* slider2 = addRow(*m_cameraObject->getApertureSizeUIInfo());
  m_MainLayout.addRow("Aperture Size", slider2);
  QNumericSlider* slider3 = addRow(*m_cameraObject->getFieldOfViewUIInfo());
  m_MainLayout.addRow("Field of view", slider3);
  QNumericSlider* slider4 = addRow(*m_cameraObject->getFocalDistanceUIInfo());
  m_MainLayout.addRow("Focal distance", slider4);

#if 0  
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

  QObject::connect(&cam->GetFilm(), SIGNAL(Changed(const QFilm&)), this, SLOT(OnFilmChanged()));
  QObject::connect(&cam->GetAperture(), SIGNAL(Changed(const QAperture&)), this, SLOT(OnApertureChanged()));
  QObject::connect(&cam->GetProjection(), SIGNAL(Changed(const QProjection&)), this, SLOT(OnProjectionChanged()));
  QObject::connect(&cam->GetFocus(), SIGNAL(Changed(const QFocus&)), this, SLOT(OnFocusChanged()));
#endif
}

void
QCameraWidget::OnFilmChanged()
{
  m_ExposureSlider.setValue(m_qcamera->GetFilm().GetExposure(), true);
  m_ExposureIterationsSpinner.blockSignals(true);
  m_ExposureIterationsSpinner.setCurrentIndex(
    m_ExposureIterationsSpinner.findData(m_qcamera->GetFilm().GetExposureIterations()));
  m_ExposureIterationsSpinner.blockSignals(false);
  m_NoiseReduction.blockSignals(true);
  m_NoiseReduction.setCheckState(m_renderSettings->m_DenoiseParams.m_Enabled ? Qt::CheckState::Checked
                                                                             : Qt::CheckState::Unchecked);
  m_NoiseReduction.blockSignals(false);
  emit m_qcamera->Changed();
}
void
QCameraWidget::OnApertureChanged()
{
  m_ApertureSizeSlider.setValue(m_qcamera->GetAperture().GetSize(), true);
  emit m_qcamera->Changed();
}
void
QCameraWidget::OnProjectionChanged()
{
  m_FieldOfViewSlider.setValue(m_qcamera->GetProjection().GetFieldOfView(), true);
  emit m_qcamera->Changed();
}
void
QCameraWidget::OnFocusChanged()
{
  m_FocalDistanceSlider.setValue(m_qcamera->GetFocus().GetFocalDistance(), true);
  emit m_qcamera->Changed();
}

QSize
QCameraWidget::sizeHint() const
{
  return QSize(20, 20);
}
