#pragma once

#include "core/prty/prtyPropertyUIInfo.hpp"

#include <string>
#include <vector>

class CheckBoxUiInfo : public prtyPropertyUIInfo
{
public:
  static constexpr const char* TYPE = "CheckBox";
  CheckBoxUiInfo(prtyProperty* i_pProperty)
    : prtyPropertyUIInfo(i_pProperty)
  {
    SetControlName(TYPE);
  }
  CheckBoxUiInfo(prtyProperty* i_pProperty, const std::string& i_Category, const std::string& i_Description)
    : prtyPropertyUIInfo(i_pProperty, i_Category, i_Description)
  {
    SetControlName(TYPE);
  }
};

class ColorPickerUiInfo : public prtyPropertyUIInfo
{
public:
  static constexpr const char* TYPE = "ColorPicker";
  ColorPickerUiInfo(prtyProperty* i_pProperty)
    : prtyPropertyUIInfo(i_pProperty)
  {
    SetControlName(TYPE);
  }
  ColorPickerUiInfo(prtyProperty* i_pProperty, const std::string& i_Category, const std::string& i_Description)
    : prtyPropertyUIInfo(i_pProperty, i_Category, i_Description)
  {
    SetControlName(TYPE);
  }
};

class ComboBoxUiInfo : public prtyPropertyUIInfo
{
public:
  static constexpr const char* TYPE = "ComboBox";
  std::vector<std::string> items;

  ComboBoxUiInfo(prtyProperty* i_pProperty)
    : prtyPropertyUIInfo(i_pProperty)
  {
    SetControlName(TYPE);
  }
  ComboBoxUiInfo(prtyProperty* i_pProperty, const std::string& i_Category, const std::string& i_Description)
    : prtyPropertyUIInfo(i_pProperty, i_Category, i_Description)
  {
    SetControlName(TYPE);
  }
};
class FloatSliderSpinnerUiInfo : public prtyPropertyUIInfo
{
public:
  static constexpr const char* TYPE = "FloatSliderSpinner";
  float min = 0.0f;
  float max = 0.0f;
  int decimals = 0;
  float singleStep = 0.0f;
  int numTickMarks = 0;
  std::string suffix;

  FloatSliderSpinnerUiInfo(prtyProperty* i_pProperty)
    : prtyPropertyUIInfo(i_pProperty)
  {
    SetControlName(TYPE);
  }
  FloatSliderSpinnerUiInfo(prtyProperty* i_pProperty, const std::string& i_Category, const std::string& i_Description)
    : prtyPropertyUIInfo(i_pProperty, i_Category, i_Description)
  {
    SetControlName(TYPE);
  }
};
//                            std::string statusTip,
//                            std::string toolTip,
//                            float min,
//                            float max,
//                            int decimals,
//                            float singleStep,
//                            int numTickMarks = 0,
//                            std::string suffix = "")
//     : GenericUIInfo(FloatSliderSpinnerUiInfo::TYPE, formLabel, statusTip, toolTip)
//     , min(min)
//     , max(max)
//     , decimals(decimals)
//     , singleStep(singleStep)
//     , numTickMarks(numTickMarks)
//     , suffix(suffix)
//   {
//   }
// };

class IntSliderSpinnerUiInfo : public prtyPropertyUIInfo
{
public:
  static constexpr const char* TYPE = "IntSliderSpinner";
  int min;
  int max;
  int singleStep;
  int numTickMarks;
  std::string suffix;

  IntSliderSpinnerUiInfo(prtyProperty* i_pProperty)
    : prtyPropertyUIInfo(i_pProperty)
  {
    SetControlName(TYPE);
  }

  IntSliderSpinnerUiInfo(prtyProperty* i_pProperty, const std::string& i_Category, const std::string& i_Description)
    : prtyPropertyUIInfo(i_pProperty, i_Category, i_Description)
  {
    SetControlName(TYPE);
  }
};
