#include "controlFactory.h"

#include "Controls.h"

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
QNumericSlider*
addRow(const IntSliderSpinnerUiInfo& info, prtyProperty<int>* prop)
{
  QNumericSlider* slider = new QNumericSlider();
  slider->setStatusTip(QString::fromStdString(info.statusTip));
  slider->setToolTip(QString::fromStdString(info.toolTip));
  slider->setRange(info.min, info.max);
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
