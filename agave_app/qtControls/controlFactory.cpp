#include "controlFactory.h"

#include "Controls.h"

QNumericSlider*
create(const FloatSliderSpinnerUiInfo* info, std::shared_ptr<prtyFloat> prop)
{
  QNumericSlider* slider = new QNumericSlider();
  slider->setStatusTip(QString::fromStdString(info->statusTip));
  slider->setToolTip(QString::fromStdString(info->toolTip));
  slider->setRange(info->min, info->max);
  slider->setDecimals(info->decimals);
  slider->setSingleStep(info->singleStep);
  slider->setNumTickMarks(info->numTickMarks);
  slider->setSuffix(QString::fromStdString(info->suffix));

  slider->setValue(prop->GetValue(), true);
  QObject::connect(
    slider, &QNumericSlider::valueChanged, [slider, prop](double value) { prop->SetValue(value, true); });
  // TODO how would this capture the "previous" value, for undo?
  // QObject::connect(slider, &QNumericSlider::valueChangeCommit, [slider, prop]() { prop->NotifyAll(true); });

  return slider;
}
QNumericSlider*
create(const IntSliderSpinnerUiInfo* info, std::shared_ptr<prtyInt32> prop)
{
  QNumericSlider* slider = new QNumericSlider();
  slider->setStatusTip(QString::fromStdString(info->statusTip));
  slider->setToolTip(QString::fromStdString(info->toolTip));
  slider->setRange(info->min, info->max);
  slider->setSingleStep(info->singleStep);
  slider->setNumTickMarks(info->numTickMarks);
  slider->setSuffix(QString::fromStdString(info->suffix));

  slider->setValue(prop->GetValue(), true);
  QObject::connect(
    slider, &QNumericSlider::valueChanged, [slider, prop](double value) { prop->SetValue(value, true); });
  // TODO how would this capture the "previous" value, for undo?
  // QObject::connect(slider, &QNumericSlider::valueChangeCommit, [slider, prop]() { prop->NotifyAll(true); });

  return slider;
}
QCheckBox*
create(const CheckBoxUiInfo* info, std::shared_ptr<prtyBoolean> prop)
{
  QCheckBox* checkBox = new QCheckBox();
  checkBox->setStatusTip(QString::fromStdString(info->statusTip));
  checkBox->setToolTip(QString::fromStdString(info->toolTip));
  // checkBox->setText(QString::fromStdString(info->formLabel));
  checkBox->setCheckState(prop->GetValue() ? Qt::CheckState::Checked : Qt::CheckState::Unchecked);
  QObject::connect(checkBox, &QCheckBox::stateChanged, [checkBox, prop](int state) {
    prop->SetValue(state == Qt::CheckState::Checked, true);
  });
  return checkBox;
}
QComboBox*
create(const ComboBoxUiInfo* info, std::shared_ptr<prtyInt8> prop)
{
  QComboBox* comboBox = new QComboBox();
  comboBox->setStatusTip(QString::fromStdString(info->statusTip));
  comboBox->setToolTip(QString::fromStdString(info->toolTip));
  for (const auto& item : info->items) {
    comboBox->addItem(QString::fromStdString(item));
  }
  comboBox->setCurrentIndex(prop->GetValue());
  QObject::connect(
    comboBox, &QComboBox::currentIndexChanged, [comboBox, prop](int index) { prop->SetValue(index, true); });
  return comboBox;
}

QNumericSlider*
addRow(const FloatSliderSpinnerUiInfo& info, prtyFloat* prop)
{
  QNumericSlider* slider = new QNumericSlider();
  slider->setStatusTip(QString::fromStdString(info.statusTip));
  slider->setToolTip(QString::fromStdString(info.toolTip));
  slider->setRange(info.min, info.max);
  slider->setDecimals(info.decimals);
  slider->setSingleStep(info.singleStep);
  slider->setNumTickMarks(info.numTickMarks);
  slider->setSuffix(QString::fromStdString(info.suffix));

  slider->setValue(prop->GetValue(), true);
  QObject::connect(
    slider, &QNumericSlider::valueChanged, [slider, prop](double value) { prop->SetValue(value, true); });
  // TODO how would this capture the "previous" value, for undo?
  // QObject::connect(slider, &QNumericSlider::valueChangeCommit, [slider, prop]() { prop->NotifyAll(true); });

  return slider;
}
QNumericSlider*
addRow(const IntSliderSpinnerUiInfo& info, prtyInt32* prop)
{
  QNumericSlider* slider = new QNumericSlider();
  slider->setStatusTip(QString::fromStdString(info.statusTip));
  slider->setToolTip(QString::fromStdString(info.toolTip));
  slider->setRange(info.min, info.max);
  slider->setSingleStep(info.singleStep);
  slider->setNumTickMarks(info.numTickMarks);
  slider->setSuffix(QString::fromStdString(info.suffix));

  slider->setValue(prop->GetValue(), true);
  QObject::connect(
    slider, &QNumericSlider::valueChanged, [slider, prop](double value) { prop->SetValue(value, true); });
  // TODO how would this capture the "previous" value, for undo?
  // QObject::connect(slider, &QNumericSlider::valueChangeCommit, [slider, prop]() { prop->NotifyAll(true); });

  return slider;
}

QComboBox*
addRow(const ComboBoxUiInfo& info, prtyInt8* prop)
{
  QComboBox* comboBox = new QComboBox();
  comboBox->setStatusTip(QString::fromStdString(info.statusTip));
  comboBox->setToolTip(QString::fromStdString(info.toolTip));
  for (const auto& item : info.items) {
    comboBox->addItem(QString::fromStdString(item));
  }
  comboBox->setCurrentIndex(prop->GetValue());
  QObject::connect(
    comboBox, &QComboBox::currentIndexChanged, [comboBox, prop](int index) { prop->SetValue(index, true); });
  return comboBox;
}

QCheckBox*
addRow(const CheckBoxUiInfo& info, prtyBoolean* prop)
{
  QCheckBox* checkBox = new QCheckBox();
  checkBox->setStatusTip(QString::fromStdString(info.statusTip));
  checkBox->setToolTip(QString::fromStdString(info.toolTip));
  // checkBox->setText(QString::fromStdString(info.formLabel));
  checkBox->setCheckState(prop->GetValue() ? Qt::CheckState::Checked : Qt::CheckState::Unchecked);
  QObject::connect(checkBox, &QCheckBox::stateChanged, [checkBox, prop](int state) {
    prop->SetValue(state == Qt::CheckState::Checked, true);
  });
  return checkBox;
}

QColorPushButton*
addRow(const ColorPickerUiInfo& info, prtyColor* prop)
{
  QColorPushButton* colorButton = new QColorPushButton();
  colorButton->setStatusTip(QString::fromStdString(info.statusTip));
  colorButton->setToolTip(QString::fromStdString(info.toolTip));
  QColor c = QColor::fromRgbF(prop->GetValue().r, prop->GetValue().g, prop->GetValue().b);
  colorButton->SetColor(c, true);
  QObject::connect(colorButton, &QColorPushButton::currentColorChanged, [colorButton, prop](const QColor& c) {
    // Convert QColor to glm::vec3
    glm::vec4 color(c.redF(), c.greenF(), c.blueF(), 1.0f);
    prop->SetValue(color, true);
  });
  return colorButton;
}