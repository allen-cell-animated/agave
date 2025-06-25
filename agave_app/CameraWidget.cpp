#include "CameraWidget.h"
#include "RenderSettings.h"

#include "qtControls/controlFactory.h"

#include "renderlib/uiInfo.hpp"
#include "renderlib/CameraUiDescription.hpp"

#include <QLabel>

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
}

QSize
QCameraWidget::sizeHint() const
{
  return QSize(20, 20);
}
