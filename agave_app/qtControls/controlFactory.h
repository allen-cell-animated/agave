#pragma once

#include "renderlib/uiInfo.hpp"
#include "renderlib/core/prty/prtyColor.hpp"
#include "renderlib/core/prty/prtyBoolean.hpp"
#include "renderlib/core/prty/prtyFloat.hpp"
#include "renderlib/core/prty/prtyInt32.hpp"
#include "renderlib/core/prty/prtyInt8.hpp"
#include "renderlib/core/prty/prtyObject.hpp"
#include "renderlib/glm.h"

#include <memory>

class QNumericSlider;
class QCheckBox;
class QComboBox;
class QColorPushButton;
class QFormLayout;
class QWidget;

QNumericSlider*
create(const FloatSliderSpinnerUiInfo* info, std::shared_ptr<prtyFloat> prop);

QNumericSlider*
create(const IntSliderSpinnerUiInfo* info, std::shared_ptr<prtyInt32> prop);

QCheckBox*
create(const CheckBoxUiInfo* info, std::shared_ptr<prtyBoolean> prop);

QComboBox*
create(const ComboBoxUiInfo* info, std::shared_ptr<prtyInt8> prop);

QNumericSlider*
addRow(const FloatSliderSpinnerUiInfo& info);

QNumericSlider*
addRow(const IntSliderSpinnerUiInfo& info);

QComboBox*
addRow(const ComboBoxUiInfo& info);

QCheckBox*
addRow(const CheckBoxUiInfo& info);

QColorPushButton*
addRow(const ColorPickerUiInfo& info);

QWidget*
addGenericRow(const prtyPropertyUIInfo& info);

void
createFlatList(QFormLayout* mainLayout, prtyObject* object);

void
createCategorizedSections(QFormLayout* mainLayout, prtyObject* object);