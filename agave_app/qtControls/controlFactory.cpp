#include "controlFactory.h"

#include "Controls.h"
#include "Section.h"

#include "renderlib/core/prty/prtyObject.hpp"
#include "renderlib/core/prty/prtyEnum.hpp"

#include <QFormLayout>

QNumericSlider*
addRow(const FloatSliderSpinnerUiInfo& info)
{
  QNumericSlider* slider = new QNumericSlider();
  slider->setStatusTip(QString::fromStdString(info.GetStatusTip()));
  slider->setToolTip(QString::fromStdString(info.GetToolTip()));
  slider->setRange(info.min, info.max);
  slider->setDecimals(info.decimals);
  slider->setSingleStep(info.singleStep);
  slider->setNumTickMarks(info.numTickMarks);
  slider->setSuffix(QString::fromStdString(info.suffix));

  auto* prop = static_cast<prtyFloat*>(info.GetProperty(0));
  slider->setValue(prop->GetValue(), true);
  auto conn = QObject::connect(
    slider, &QNumericSlider::valueChanged, [slider, prop](double value) { prop->SetValue(value, true); });
  // now add a callback to the property to update the control when the property changes

  //	Note: right now, this will not create a circular update because
  //	of the m_bLocalChangeNoUpdate flag.  This IS NOT true the other way
  //	around.  If a control calls its "ValueChanged" then the property will
  //	call all of its callback controls and one of them could have been the one
  //	that originally updated the property value(s).  By checking the diff of
  //	the values we can avoid a repetitive setting of the control's values.
  prop->AddCallback(new prtyCallbackLambda([slider](prtyProperty* i_pProperty, bool i_bDirty) {
    // this is equivalent to QWidget::blockSignals(true);
    // if (m_bLocalChangeNoUpdate)
    //  return;
    const float newvalue = (static_cast<prtyFloat*>(i_pProperty))->GetValue();
    // m_bLocalChangeNoUpdate = true;
    slider->blockSignals(true);
    if ((float)slider->value() != newvalue) {
      // Prevent recursive updates
      // slider->setLocalChangeNoUpdate(true);
      slider->setValue(newvalue, false);
      // slider->setLocalChangeNoUpdate(false);
    }
    // m_bLocalChangeNoUpdate = false;
    slider->blockSignals(false);
  }));

  QObject::connect(slider, &QNumericSlider::destroyed, [conn]() {
    // Disconnect the signal when the slider is destroyed
    QObject::disconnect(conn);
  });
  return slider;
}

QNumericSlider*
addRow(const IntSliderSpinnerUiInfo& info)
{
  QNumericSlider* slider = new QNumericSlider();
  slider->setStatusTip(QString::fromStdString(info.GetStatusTip()));
  slider->setToolTip(QString::fromStdString(info.GetToolTip()));
  slider->setRange(info.min, info.max);
  slider->setSingleStep(info.singleStep);
  slider->setNumTickMarks(info.numTickMarks);
  slider->setSuffix(QString::fromStdString(info.suffix));

  auto* prop = static_cast<prtyInt32*>(info.GetProperty(0));
  slider->setValue(prop->GetValue(), true);
  auto conn = QObject::connect(
    slider, &QNumericSlider::valueChanged, [slider, prop](double value) { prop->SetValue(value, true); });
  QObject::connect(slider, &QNumericSlider::destroyed, [conn]() {
    // Disconnect the signal when the slider is destroyed
    QObject::disconnect(conn);
  });
  return slider;
}

QComboBox*
addRow(const ComboBoxUiInfo& info)
{
  QComboBox* comboBox = new QComboBox();
  comboBox->setStatusTip(QString::fromStdString(info.GetStatusTip()));
  comboBox->setToolTip(QString::fromStdString(info.GetToolTip()));
  auto* prop = static_cast<prtyEnum*>(info.GetProperty(0));
  for (int i = 0; i < prop->GetNumTags(); ++i) {
    comboBox->addItem(QString::fromStdString(prop->GetEnumTag(i)));
  }
  comboBox->setCurrentIndex(prop->GetValue());
  auto conn = QObject::connect(
    comboBox, &QComboBox::currentIndexChanged, [comboBox, prop](int index) { prop->SetValue(index, true); });
  QObject::connect(comboBox, &QComboBox::destroyed, [conn]() {
    // Disconnect the signal when the combobox is destroyed
    QObject::disconnect(conn);
  });
  return comboBox;
}

QCheckBox*
addRow(const CheckBoxUiInfo& info)
{
  QCheckBox* checkBox = new QCheckBox();
  checkBox->setStatusTip(QString::fromStdString(info.GetStatusTip()));
  checkBox->setToolTip(QString::fromStdString(info.GetToolTip()));
  auto* prop = static_cast<prtyBoolean*>(info.GetProperty(0));
  checkBox->setCheckState(prop->GetValue() ? Qt::CheckState::Checked : Qt::CheckState::Unchecked);
  auto conn = QObject::connect(checkBox, &QCheckBox::stateChanged, [checkBox, prop](int state) {
    prop->SetValue(state == Qt::CheckState::Checked, true);
  });
  QObject::connect(checkBox, &QCheckBox::destroyed, [conn]() {
    // Disconnect the signal when the checkbox is destroyed
    QObject::disconnect(conn);
  });
  return checkBox;
}

QColorPushButton*
addRow(const ColorPickerUiInfo& info)
{
  QColorPushButton* colorButton = new QColorPushButton();
  colorButton->setStatusTip(QString::fromStdString(info.GetStatusTip()));
  colorButton->setToolTip(QString::fromStdString(info.GetToolTip()));
  auto* prop = static_cast<prtyColor*>(info.GetProperty(0));
  QColor c = QColor::fromRgbF(prop->GetValue().r, prop->GetValue().g, prop->GetValue().b);
  colorButton->SetColor(c, true);
  auto conn =
    QObject::connect(colorButton, &QColorPushButton::currentColorChanged, [colorButton, prop](const QColor& c) {
      // Convert QColor to glm::vec3
      glm::vec4 color(c.redF(), c.greenF(), c.blueF(), 1.0f);
      prop->SetValue(color, true);
    });
  QObject::connect(colorButton, &QColorPushButton::destroyed, [conn]() {
    // Disconnect the signal when the button is destroyed
    QObject::disconnect(conn);
  });
  return colorButton;
}

QWidget*
addGenericRow(const prtyPropertyUIInfo& info)
{
  // TODO: consider checking info.GetControlName() to determine the type,
  // but using dynamic_cast is more type-safe and extensible.
  // If a new type is added, it will require a new dynamic_cast case here.
  if (const auto* floatInfo = dynamic_cast<const FloatSliderSpinnerUiInfo*>(&info)) {
    return addRow(*floatInfo);
  } else if (const auto* intInfo = dynamic_cast<const IntSliderSpinnerUiInfo*>(&info)) {
    return addRow(*intInfo);
  } else if (const auto* comboBoxInfo = dynamic_cast<const ComboBoxUiInfo*>(&info)) {
    return addRow(*comboBoxInfo);
  } else if (const auto* checkBoxInfo = dynamic_cast<const CheckBoxUiInfo*>(&info)) {
    return addRow(*checkBoxInfo);
  } else if (const auto* colorPickerInfo = dynamic_cast<const ColorPickerUiInfo*>(&info)) {
    return addRow(*colorPickerInfo);
  }

  return nullptr; // or throw an exception
}

void
createCategorizedSections(QFormLayout* mainLayout, prtyObject* object)
{
  // Map to organize properties by category
  std::map<std::string, std::vector<std::shared_ptr<prtyPropertyUIInfo>>> categorizedProperties;

  // Group properties by category
  const auto& propertyList = object->GetList();
  for (const auto& propertyInfo : propertyList) {
    if (propertyInfo) {
      std::string category = propertyInfo->GetCategory();
      categorizedProperties[category].push_back(propertyInfo);
    }
  }

  // Create sections for each category (automatically sorted by std::map)
  for (const auto& [category, properties] : categorizedProperties) {
    if (!properties.empty()) {
      // Create section
      Section* section = new Section(QString::fromStdString(category));

      // Create form layout for this section's content
      QFormLayout* sectionLayout = new QFormLayout();

      // Add controls for each property in this category
      for (const auto& propertyInfo : properties) {
        QWidget* control = addGenericRow(*propertyInfo);
        if (control) {
          QString label = QString::fromStdString(propertyInfo->GetDescription());
          sectionLayout->addRow(label, control);
        }
      }

      // Set the section's content layout
      section->setContentLayout(*sectionLayout);

      // Add section to main layout
      mainLayout->addRow(section);
    }
  }
}

void
createFlatList(QFormLayout* mainLayout, prtyObject* object)
{
  // Simple flat list of all properties
  const auto& propertyList = object->GetList();
  for (const auto& propertyInfo : propertyList) {
    if (propertyInfo) {
      QWidget* control = addGenericRow(*propertyInfo);
      if (control) {
        QString label = QString::fromStdString(propertyInfo->GetDescription());
        mainLayout->addRow(label, control);
      }
    }
  }
}
