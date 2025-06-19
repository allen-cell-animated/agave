#pragma once

#include "renderlib/uiInfo.hpp"
#include "renderlib/core/prty/prtyColor.hpp"
#include "renderlib/core/prty/prtyBoolean.hpp"
#include "renderlib/core/prty/prtyFloat.hpp"
#include "renderlib/core/prty/prtyInt32.hpp"
#include "renderlib/core/prty/prtyInt8.hpp"
#include "renderlib/glm.h"

#include <memory>

class QNumericSlider;
class QCheckBox;
class QComboBox;
class QColorPushButton;

QNumericSlider*
create(const FloatSliderSpinnerUiInfo* info, std::shared_ptr<prtyFloat> prop);

QNumericSlider*
create(const IntSliderSpinnerUiInfo* info, std::shared_ptr<prtyInt32> prop);

QCheckBox*
create(const CheckBoxUiInfo* info, std::shared_ptr<prtyBoolean> prop);

QComboBox*
create(const ComboBoxUiInfo* info, std::shared_ptr<prtyInt8> prop);

QNumericSlider*
addRow(const FloatSliderSpinnerUiInfo& info, prtyFloat* prop);

QNumericSlider*
addRow(const IntSliderSpinnerUiInfo& info, prtyInt32* prop);

QComboBox*
addRow(const ComboBoxUiInfo& info, prtyInt8* prop);

QCheckBox*
addRow(const CheckBoxUiInfo& info, prtyBoolean* prop);

QColorPushButton*
addRow(const ColorPickerUiInfo& info, prtyColor* prop);
