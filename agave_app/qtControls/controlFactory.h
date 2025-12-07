#pragma once

#include "renderlib/uiInfo.hpp"
#include "renderlib/core/prty/prtyColor.hpp"
#include "renderlib/core/prty/prtyBoolean.hpp"
#include "renderlib/core/prty/prtyEnum.hpp"
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
addRow(const FloatSliderSpinnerUiInfo& info);

QNumericSlider*
addRow(const IntSliderSpinnerUiInfo& info);

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

template<typename LayoutType>
void
createFlatList(LayoutType* mainLayout, prtyObject* object);

void
createCategorizedSections(QFormLayout* mainLayout, prtyObject* object);
