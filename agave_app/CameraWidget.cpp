#include "CameraWidget.h"
#include "RenderSettings.h"

#include "renderlib/uiInfo.hpp"

#include <QLabel>

QNumericSlider*
create(const FloatSliderSpinnerUiInfo* info, std::shared_ptr<prtyProperty<float>> prop)
{
  QNumericSlider* slider = new QNumericSlider();
  slider->setStatusTip(QString::fromStdString(info->statusTip));
  slider->setToolTip(QString::fromStdString(info->toolTip));
  slider->setRange(info->min, info->max);
  slider->setDecimals(info->decimals);
  slider->setSingleStep(info->singleStep);
  slider->setNumTickMarks(info->numTickMarks);
  slider->setSuffix(QString::fromStdString(info->suffix));

  slider->setValue(prop->get(), true);
  QObject::connect(slider, &QNumericSlider::valueChanged, [slider, prop](double value) { prop->set(value, true); });
  // TODO how would this capture the "previous" value, for undo?
  QObject::connect(slider, &QNumericSlider::valueChangeCommit, [slider, prop]() { prop->notifyAll(true); });

  return slider;
}
QNumericSlider*
create(const IntSliderSpinnerUiInfo* info, std::shared_ptr<prtyProperty<int>> prop)
{
  QNumericSlider* slider = new QNumericSlider();
  slider->setStatusTip(QString::fromStdString(info->statusTip));
  slider->setToolTip(QString::fromStdString(info->toolTip));
  slider->setRange(info->min, info->max);
  slider->setSingleStep(info->singleStep);
  slider->setNumTickMarks(info->numTickMarks);
  slider->setSuffix(QString::fromStdString(info->suffix));

  slider->setValue(prop->get(), true);
  QObject::connect(slider, &QNumericSlider::valueChanged, [slider, prop](double value) { prop->set(value, true); });
  // TODO how would this capture the "previous" value, for undo?
  QObject::connect(slider, &QNumericSlider::valueChangeCommit, [slider, prop]() { prop->notifyAll(true); });

  return slider;
}
QCheckBox*
create(const CheckBoxUiInfo* info, std::shared_ptr<prtyProperty<bool>> prop)
{
  QCheckBox* checkBox = new QCheckBox();
  checkBox->setStatusTip(QString::fromStdString(info->statusTip));
  checkBox->setToolTip(QString::fromStdString(info->toolTip));
  // checkBox->setText(QString::fromStdString(info->formLabel));
  checkBox->setCheckState(prop->get() ? Qt::CheckState::Checked : Qt::CheckState::Unchecked);
  QObject::connect(checkBox, &QCheckBox::stateChanged, [checkBox, prop](int state) {
    prop->set(state == Qt::CheckState::Checked, true);
  });
  return checkBox;
}
QComboBox*
create(const ComboBoxUiInfo* info, std::shared_ptr<prtyProperty<int>> prop)
{
  QComboBox* comboBox = new QComboBox();
  comboBox->setStatusTip(QString::fromStdString(info->statusTip));
  comboBox->setToolTip(QString::fromStdString(info->toolTip));
  for (const auto& item : info->items) {
    comboBox->addItem(QString::fromStdString(item));
  }
  comboBox->setCurrentIndex(prop->get());
  QObject::connect(comboBox, &QComboBox::currentIndexChanged, [comboBox, prop](int index) { prop->set(index, true); });
  return comboBox;
}

QNumericSlider*
addRow(const FloatSliderSpinnerUiInfo& info, prtyProperty<float>* prop)
{
  QNumericSlider* slider = new QNumericSlider();
  slider->setStatusTip(QString::fromStdString(info.statusTip));
  slider->setToolTip(QString::fromStdString(info.toolTip));
  slider->setRange(info.min, info.max);
  slider->setDecimals(info.decimals);
  slider->setSingleStep(info.singleStep);
  slider->setNumTickMarks(info.numTickMarks);
  slider->setSuffix(QString::fromStdString(info.suffix));

  slider->setValue(prop->get(), true);
  QObject::connect(slider, &QNumericSlider::valueChanged, [slider, prop](double value) { prop->set(value, true); });
  // TODO how would this capture the "previous" value, for undo?
  QObject::connect(slider, &QNumericSlider::valueChangeCommit, [slider, prop]() { prop->notifyAll(true); });

  return slider;
}
QComboBox*
addRow(const ComboBoxUiInfo& info, prtyProperty<int>* prop)
{
  QComboBox* comboBox = new QComboBox();
  comboBox->setStatusTip(QString::fromStdString(info.statusTip));
  comboBox->setToolTip(QString::fromStdString(info.toolTip));
  for (const auto& item : info.items) {
    comboBox->addItem(QString::fromStdString(item));
  }
  comboBox->setCurrentIndex(prop->get());
  QObject::connect(comboBox, &QComboBox::currentIndexChanged, [comboBox, prop](int index) { prop->set(index, true); });
  return comboBox;
}
QCheckBox*
addRow(const CheckBoxUiInfo& info, prtyProperty<bool>* prop)
{
  QCheckBox* checkBox = new QCheckBox();
  checkBox->setStatusTip(QString::fromStdString(info.statusTip));
  checkBox->setToolTip(QString::fromStdString(info.toolTip));
  // checkBox->setText(QString::fromStdString(info.formLabel));
  checkBox->setCheckState(prop->get() ? Qt::CheckState::Checked : Qt::CheckState::Unchecked);
  QObject::connect(checkBox, &QCheckBox::stateChanged, [checkBox, prop](int state) {
    prop->set(state == Qt::CheckState::Checked, true);
  });
  return checkBox;
}

QCameraWidget::QCameraWidget(QWidget* pParent, QCamera* cam, RenderSettings* rs, CameraDataObject* cdo)
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
