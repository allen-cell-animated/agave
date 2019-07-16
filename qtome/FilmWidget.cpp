#include "FilmWidget.h"
#include "Camera.h"
#include "renderlib/RenderSettings.h"

#include <QLabel>
#include <QVariant>

#include <math.h>

QFilmWidget::QFilmWidget(QWidget* pParent, QCamera* cam, RenderSettings* rs)
  : QGroupBox(pParent)
  , m_Layout()
  , m_PresetType()
  , m_PresetsLayout()
  , m_WidthSpinner()
  , m_HeightSpinner()
  , m_ExposureSlider()
  , m_ExposureIterationsSpinner()
  , m_NoiseReduction()
  , m_qcamera(cam)
  , m_renderSettings(rs)
{
  setTitle("Film");
  setStatusTip("Film properties");
  setToolTip("Film properties");

  // Create grid layout
  setLayout(&m_Layout);

  const int ResMin = powf(2.0f, 5);
  const int ResMax = powf(2.0f, 11);

  // Exposure

  m_ExposureSlider.setRange(0.0f, 1.0f);
  m_ExposureSlider.setValue(cam->GetFilm().GetExposure());
  m_Layout.addRow("Exposure", &m_ExposureSlider);

  QObject::connect(&m_ExposureSlider, SIGNAL(valueChanged(double)), this, SLOT(SetExposure(double)));

  // Exposure
  m_ExposureIterationsSpinner.addItem("1", 1);
  m_ExposureIterationsSpinner.addItem("2", 2);
  m_ExposureIterationsSpinner.addItem("4", 4);
  m_ExposureIterationsSpinner.addItem("8", 8);
  m_ExposureIterationsSpinner.setCurrentIndex(
    m_ExposureIterationsSpinner.findData(cam->GetFilm().GetExposureIterations()));
  m_Layout.addRow("Exposure Time", &m_ExposureIterationsSpinner);
  QObject::connect(&m_ExposureIterationsSpinner,
                   SIGNAL(currentIndexChanged(const QString&)),
                   this,
                   SLOT(SetExposureIterations(const QString&)));

  // gStatus.SetStatisticChanged("Camera", "Film", "", "", "");

  // m_NoiseReduction.setText("Noise Reduction");
  m_NoiseReduction.setCheckState(rs->m_DenoiseParams.m_Enabled ? Qt::CheckState::Checked : Qt::CheckState::Unchecked);
  m_Layout.addRow("Noise Reduction", &m_NoiseReduction);

  QObject::connect(&m_NoiseReduction, SIGNAL(stateChanged(const int&)), this, SLOT(OnNoiseReduction(const int&)));

  // QObject::connect(&gStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
  // QObject::connect(&gStatus, SIGNAL(RenderEnd()), this, SLOT(OnRenderEnd()));

  QObject::connect(&cam->GetFilm(), SIGNAL(Changed(const QFilm&)), this, SLOT(OnFilmChanged(const QFilm&)));

  //	OnRenderBegin();
}

void
QFilmWidget::SetPresetType(const QString& PresetType)
{
  if (PresetType == "NTSC D-1 (video)") {
    m_Preset[0].SetPreset(720, 486);
    m_Preset[1].SetPreset(200, 135);
    m_Preset[2].SetPreset(360, 243);
    m_Preset[3].SetPreset(512, 346);
  }

  if (PresetType == "NTSC DV (video)") {
    m_Preset[0].SetPreset(720, 480);
    m_Preset[1].SetPreset(300, 200);
    m_Preset[2].SetPreset(360, 240);
    m_Preset[3].SetPreset(512, 341);
  }

  if (PresetType == "PAL (video)") {
    m_Preset[0].SetPreset(768, 576);
    m_Preset[1].SetPreset(180, 135);
    m_Preset[2].SetPreset(240, 180);
    m_Preset[3].SetPreset(480, 360);
  }

  if (PresetType == "PAL D-1 (video)") {
    m_Preset[0].SetPreset(720, 576);
    m_Preset[1].SetPreset(180, 144);
    m_Preset[2].SetPreset(240, 192);
    m_Preset[3].SetPreset(480, 384);
  }

  if (PresetType == "HDTV (video)") {
    m_Preset[0].SetPreset(1920, 1080);
    m_Preset[1].SetPreset(490, 270);
    m_Preset[2].SetPreset(1280, 720);
    m_Preset[3].SetPreset(320, 180);
  }
}

void
QFilmWidget::SetPreset(QFilmResolutionPreset& Preset)
{
  m_WidthSpinner.setValue(Preset.GetWidth());
  m_HeightSpinner.setValue(Preset.GetHeight());
}

void
QFilmWidget::SetWidth(const int& Width)
{
  m_qcamera->GetFilm().SetWidth(Width);
}

void
QFilmWidget::SetHeight(const int& Height)
{
  m_qcamera->GetFilm().SetHeight(Height);
}

void
QFilmWidget::SetExposure(const double& Exposure)
{
  m_qcamera->GetFilm().SetExposure(Exposure);
}

void
QFilmWidget::SetExposureIterations(const QString& ExposureIterations)
{
  int value = m_ExposureIterationsSpinner.currentData().toInt();
  m_qcamera->GetFilm().SetExposureIterations(value);
}

void
QFilmWidget::OnRenderBegin(void)
{
  m_WidthSpinner.setValue(m_qcamera->GetFilm().GetWidth());
  m_HeightSpinner.setValue(m_qcamera->GetFilm().GetHeight());
  m_ExposureSlider.setValue(m_qcamera->GetFilm().GetExposure());
  m_ExposureIterationsSpinner.setCurrentIndex(
    m_ExposureIterationsSpinner.findData(m_qcamera->GetFilm().GetExposureIterations()));

  m_NoiseReduction.setChecked(m_renderSettings->m_DenoiseParams.m_Enabled);
}

void
QFilmWidget::OnRenderEnd(void)
{}

void
QFilmWidget::OnFilmChanged(const QFilm& Film)
{
  // Width
  m_WidthSpinner.blockSignals(true);
  m_WidthSpinner.setValue(Film.GetWidth());
  m_WidthSpinner.blockSignals(false);

  // Height
  m_HeightSpinner.blockSignals(true);
  m_HeightSpinner.setValue(Film.GetHeight());
  m_HeightSpinner.blockSignals(false);

  // Exposure
  m_ExposureSlider.setValue(Film.GetExposure(), true);

  m_ExposureIterationsSpinner.blockSignals(true);
  m_ExposureIterationsSpinner.setCurrentIndex(m_ExposureIterationsSpinner.findData(Film.GetExposureIterations()));
  m_ExposureIterationsSpinner.blockSignals(false);
}

void
QFilmWidget::OnNoiseReduction(const int& ReduceNoise)
{
  m_qcamera->GetFilm().SetNoiseReduction(m_NoiseReduction.checkState());
}
