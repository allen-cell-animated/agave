#pragma once

#include <string>
#include <vector>

struct GenericUIInfo
{
  std::string type;
  std::string formLabel;
  std::string statusTip;
  std::string toolTip;

  GenericUIInfo() = default;
  GenericUIInfo(std::string type, std::string formLabel, std::string statusTip, std::string toolTip)
    : type(type)
    , formLabel(formLabel)
    , statusTip(statusTip)
    , toolTip(toolTip)
  {
  }
};
struct CheckBoxUiInfo : public GenericUIInfo
{
  static constexpr const char* TYPE = "CheckBox";

  CheckBoxUiInfo() { type = CheckBoxUiInfo::TYPE; }
  CheckBoxUiInfo(std::string formLabel, std::string statusTip, std::string toolTip)
    : GenericUIInfo(CheckBoxUiInfo::TYPE, formLabel, statusTip, toolTip)
  {
  }
};
struct ComboBoxUiInfo : public GenericUIInfo
{
  static constexpr const char* TYPE = "ComboBox";
  std::vector<std::string> items;

  ComboBoxUiInfo() { type = ComboBoxUiInfo::TYPE; }
  ComboBoxUiInfo(std::string formLabel, std::string statusTip, std::string toolTip, std::vector<std::string> items)
    : GenericUIInfo(ComboBoxUiInfo::TYPE, formLabel, statusTip, toolTip)
    , items(items)
  {
  }
};

class ColorWithIntensityUiInfo : public prtyPropertyUIInfo
{
public:
  static constexpr const char* TYPE = "ColorWithIntensity";
  float min = 0.0f;
  float max = 0.0f;
  int decimals = 0;
  float singleStep = 0.0f;
  int numTickMarks = 0;
  std::string suffix;

  ColorWithIntensityUiInfo(prtyProperty* i_pColorProperty, prtyProperty* i_pIntensityProperty)
    : prtyPropertyUIInfo(i_pColorProperty)
  {
    AddProperty(i_pIntensityProperty);
    SetControlName(TYPE);
  }
  ColorWithIntensityUiInfo(prtyProperty* i_pColorProperty,
                           prtyProperty* i_pIntensityProperty,
                           const std::string& i_Category,
                           const std::string& i_Description)
    : prtyPropertyUIInfo(i_pColorProperty, i_Category, i_Description)
  {
    AddProperty(i_pIntensityProperty);
    SetControlName(TYPE);
  }
};

class FloatSliderSpinnerUiInfo : public prtyPropertyUIInfo
{
  static constexpr const char* TYPE = "FloatSliderSpinner";
  float min = 0.0f;
  float max = 0.0f;
  int decimals = 0;
  float singleStep = 0.0f;
  int numTickMarks = 0;
  std::string suffix;

  FloatSliderSpinnerUiInfo() { type = FloatSliderSpinnerUiInfo::TYPE; }
  FloatSliderSpinnerUiInfo(std::string formLabel,
                           std::string statusTip,
                           std::string toolTip,
                           float min,
                           float max,
                           int decimals,
                           float singleStep,
                           int numTickMarks = 0,
                           std::string suffix = "")
    : GenericUIInfo(FloatSliderSpinnerUiInfo::TYPE, formLabel, statusTip, toolTip)
    , min(min)
    , max(max)
    , decimals(decimals)
    , singleStep(singleStep)
    , numTickMarks(numTickMarks)
    , suffix(suffix)
  {
  }
};
struct IntSliderSpinnerUiInfo : public GenericUIInfo
{
  static constexpr const char* TYPE = "IntSliderSpinner";
  int min;
  int max;
  int singleStep;
  int numTickMarks;
  std::string suffix;

  IntSliderSpinnerUiInfo() { type = IntSliderSpinnerUiInfo::TYPE; }
};

struct ColorPickerUiInfo : public GenericUIInfo
{
  static constexpr const char* TYPE = "ColorPicker";

  ColorPickerUiInfo() { type = ColorPickerUiInfo::TYPE; }
  ColorPickerUiInfo(std::string formLabel, std::string statusTip, std::string toolTip)
    : GenericUIInfo(ColorPickerUiInfo::TYPE, formLabel, statusTip, toolTip)
  {
  }
};
