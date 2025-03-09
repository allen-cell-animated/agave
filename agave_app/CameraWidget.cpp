#include "CameraWidget.h"
#include "RenderSettings.h"

#include "qtControls/controlFactory.h"

#include "renderlib/uiInfo.hpp"
#include "renderlib/CameraUiDescription.hpp"

#include <QLabel>

QCameraWidget::QCameraWidget(QWidget* pParent, RenderSettings* rs, CameraDataObject* cdo)
  : QWidget(pParent)
  , m_MainLayout()
  , m_renderSettings(rs)
  , m_cameraDataObject(cdo)
{
  Controls::initFormLayout(m_MainLayout);
  setLayout(&m_MainLayout);

  QNumericSlider* slider = addRow(CameraUiDescription::m_exposure, &m_cameraDataObject->Exposure);
  m_MainLayout.addRow("Exposure", slider);
  QComboBox* comboBox = addRow(CameraUiDescription::m_exposureIterations, &m_cameraDataObject->ExposureIterations);
  m_MainLayout.addRow("Exposure Time", comboBox);
  QCheckBox* checkBox = addRow(CameraUiDescription::m_noiseReduction, &m_cameraDataObject->NoiseReduction);
  m_MainLayout.addRow("Noise Reduction", checkBox);
  QNumericSlider* slider2 = addRow(CameraUiDescription::m_apertureSize, &m_cameraDataObject->ApertureSize);
  m_MainLayout.addRow("Aperture Size", slider2);
  QNumericSlider* slider3 = addRow(CameraUiDescription::m_fieldOfView, &m_cameraDataObject->FieldOfView);
  m_MainLayout.addRow("Field of view", slider3);
  QNumericSlider* slider4 = addRow(CameraUiDescription::m_focalDistance, &m_cameraDataObject->FocalDistance);
  m_MainLayout.addRow("Focal distance", slider4);
}

QSize
QCameraWidget::sizeHint() const
{
  return QSize(20, 20);
}
