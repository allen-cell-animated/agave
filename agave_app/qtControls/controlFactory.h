#pragma once

#include "renderlib/uiInfo.hpp"
#include "renderlib/core/prty/prtyProperty.h"

#include <memory>

class QNumericSlider;
class QCheckBox;
class QComboBox;

QNumericSlider*
create(const FloatSliderSpinnerUiInfo* info, std::shared_ptr<prtyProperty<float>> prop);

QNumericSlider*
create(const IntSliderSpinnerUiInfo* info, std::shared_ptr<prtyProperty<int>> prop);

QCheckBox*
create(const CheckBoxUiInfo* info, std::shared_ptr<prtyProperty<bool>> prop);

QComboBox*
create(const ComboBoxUiInfo* info, std::shared_ptr<prtyProperty<int>> prop);

QNumericSlider*
addRow(const FloatSliderSpinnerUiInfo& info, prtyProperty<float>* prop);

QComboBox*
addRow(const ComboBoxUiInfo& info, prtyProperty<int>* prop);

QCheckBox*
addRow(const CheckBoxUiInfo& info, prtyProperty<bool>* prop);