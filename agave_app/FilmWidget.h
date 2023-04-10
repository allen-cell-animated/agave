#pragma once

#include <QCheckBox>
#include <QGroupBox>
#include <QSpinBox>

#include "Controls.h"
#include "Film.h"

class RenderSettings;
class QCamera;

class QFilmResolutionPreset
{
public:
  QFilmResolutionPreset(void)
    : m_Width(0)
    , m_Height(0)
  {
  }

  QFilmResolutionPreset(const int& Width, const int& Height)
    : m_Width(Width)
    , m_Height(Height)
  {
  }

  int GetWidth(void) const { return m_Width; }
  void SetWidth(const int& Width) { m_Width = Width; }
  int GetHeight(void) const { return m_Height; }
  void SetHeight(const int& Height) { m_Height = Height; }

private:
  int m_Width;
  int m_Height;
};

class QFilmResolutionButton : public QPushButton
{
  Q_OBJECT

public:
  QFilmResolutionButton(void){};

  QFilmResolutionButton(const int& Width, const int& Height)
    : m_Preset(Width, Height)
  {
  }

  void SetPreset(const int& Width, const int& Height)
  {
    m_Preset.SetWidth(Width);
    m_Preset.SetHeight(Height);

    const QString Message = QString::number(Width) + " x " + QString::number(Height);

    setText(Message);
    setToolTip(Message);
    setStatusTip("Change render resolution to " + Message);
  }

  void mousePressEvent(QMouseEvent* pEvent) { emit SetPreset(m_Preset); }

signals:
  void SetPreset(QFilmResolutionPreset& Preset);

private:
  QFilmResolutionPreset m_Preset;
};

class QFilmWidget : public QGroupBox
{
  Q_OBJECT

public:
  QFilmWidget(QWidget* pParent = NULL, QCamera* cam = nullptr, RenderSettings* rs = nullptr);

public slots:
  void SetPresetType(const QString& PresetType);
  void SetPreset(QFilmResolutionPreset& Preset);
  void SetWidth(const int& Width);
  void SetHeight(const int& Height);
  void SetExposure(const double& Exposure);
  void SetExposureIterations(const QString& ExposureIterations);
  void OnRenderBegin(void);
  void OnRenderEnd(void);
  void OnFilmChanged(const QFilm& Film);
  void OnNoiseReduction(const int& ReduceNoise);

private:
  QFormLayout m_Layout;
  QComboBox m_PresetType;
  QGridLayout m_PresetsLayout;
  QFilmResolutionButton m_Preset[4];
  QSpinBox m_WidthSpinner;
  QSpinBox m_HeightSpinner;
  QNumericSlider m_ExposureSlider;
  QComboBox m_ExposureIterationsSpinner;
  QCheckBox m_NoiseReduction;

  QCamera* m_qcamera;
  RenderSettings* m_renderSettings;
};
