#include "CameraWidget.h"
#include "RenderSettings.h"

#include "qtControls/controlFactory.h"

#include "renderlib/uiInfo.hpp"

#include <QLabel>

QCameraWidget::QCameraWidget(QWidget* pParent, RenderSettings* rs, CameraDataObject* cdo)
  : QWidget(pParent)
  , m_MainLayout()
  , m_renderSettings(rs)
  , m_cameraDataObject(cdo)
{
  Controls::initFormLayout(m_MainLayout);
  setLayout(&m_MainLayout);

  QNumericSlider* slider = addRow(FloatSliderSpinnerUiInfo("Exposure",
                                                           "Set Exposure",
                                                           "Set camera exposure",
                                                           0.0f,
                                                           1.0f,
                                                           2,    // decimals
                                                           0.01, // singleStep
                                                           0     // numTickMarks
                                                           ),
                                  &m_cameraDataObject->Exposure);
  m_MainLayout.addRow("Exposure", slider);
  QComboBox* comboBox = addRow(ComboBoxUiInfo("Exposure Time",
                                              "Set Exposure Time",
                                              "Set number of samples to accumulate per viewport update",
                                              { "1", "2", "4", "8" }),
                               &m_cameraDataObject->ExposureIterations);
  m_MainLayout.addRow("Exposure Time", comboBox);
  QCheckBox* checkBox = addRow(CheckBoxUiInfo("Noise Reduction", "Enable denoising pass", "Enable denoising pass"),
                               &m_cameraDataObject->NoiseReduction);
  m_MainLayout.addRow("Noise Reduction", checkBox);
  QNumericSlider* slider2 = addRow(FloatSliderSpinnerUiInfo("Aperture Size",
                                                            "Set camera aperture size",
                                                            "Set camera aperture size",
                                                            0.0f,
                                                            0.1f,
                                                            2,    // decimals
                                                            0.01, // singleStep
                                                            0,    // numTickMarks
                                                            " mm"),
                                   &m_cameraDataObject->ApertureSize);
  m_MainLayout.addRow("Aperture Size", slider2);
  QNumericSlider* slider3 = addRow(FloatSliderSpinnerUiInfo("Field of view",
                                                            "Set camera field of view angle",
                                                            "Set camera field of view angle",
                                                            10.0f,
                                                            150.0f,
                                                            2,    // decimals
                                                            0.01, // singleStep
                                                            0,    // numTickMarks
                                                            " deg."),
                                   &m_cameraDataObject->FieldOfView);
  m_MainLayout.addRow("Field of view", slider3);
  QNumericSlider* slider4 = addRow(FloatSliderSpinnerUiInfo("Focal distance",
                                                            "Set focal distance",
                                                            "Set focal distance",
                                                            0.0f,
                                                            15.0f,
                                                            2,    // decimals
                                                            0.01, // singleStep
                                                            0,    // numTickMarks
                                                            " m"),
                                   &m_cameraDataObject->FocalDistance);
  m_MainLayout.addRow("Focal distance", slider4);
}

QSize
QCameraWidget::sizeHint() const
{
  return QSize(20, 20);
}
