#pragma once

#include "Colormap.h"
#include "GradientData.h"
#include "core/prty/prtyColor.hpp"
#include "core/prty/prtyFloat.hpp"
#include "core/prty/prtyEnum.hpp"
#include "core/prty/prtyBoolean.hpp"
#include "core/prty/prtyVector3d.hpp"
#include "core/prty/prtyText.hpp"

class prtyControlPointVector
  : public prtyPropertyTemplate<std::vector<LutControlPoint>, const std::vector<LutControlPoint>&>
{
public:
  prtyControlPointVector(const std::string& i_Name)
    : prtyPropertyTemplate<std::vector<LutControlPoint>, const std::vector<LutControlPoint>&>(
        i_Name,
        std::vector<LutControlPoint>{ { 0.0f, 0.0f }, { 1.0f, 1.0f } })
  {
  }
  prtyControlPointVector(const std::string& i_Name, const std::vector<LutControlPoint>& i_InitialValue)
    : prtyPropertyTemplate<std::vector<LutControlPoint>, const std::vector<LutControlPoint>&>(i_Name, i_InitialValue)
  {
  }

  virtual const char* GetType() override { return "ControlPointVector"; }
  virtual void Read(docReader& io_Reader) override
  {
    // Implement reading from a reader if needed
  }
  virtual void Write(docWriter& io_Writer) const override
  {
    // Implement writing to a writer if needed
  }
};

class prtyColorControlPointVector
  : public prtyPropertyTemplate<std::vector<ColorControlPoint>, const std::vector<ColorControlPoint>&>
{
public:
  prtyColorControlPointVector(const std::string& i_Name)
    : prtyPropertyTemplate<std::vector<ColorControlPoint>, const std::vector<ColorControlPoint>&>(
        i_Name,
        std::vector<ColorControlPoint>{ { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }, { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f } })
  {
  }
  prtyColorControlPointVector(const std::string& i_Name, const std::vector<ColorControlPoint>& i_InitialValue)
    : prtyPropertyTemplate<std::vector<ColorControlPoint>, const std::vector<ColorControlPoint>&>(i_Name,
                                                                                                  i_InitialValue)
  {
  }

  virtual const char* GetType() override { return "ColorControlPointVector"; }
  virtual void Read(docReader& io_Reader) override
  {
    // Implement reading from a reader if needed
  }
  virtual void Write(docWriter& io_Writer) const override
  {
    // Implement writing to a writer if needed
  }
};

class GradientDataObject
{
public:
  GradientDataObject() = default;

  // Gradient related properties can be added here
  prtyEnum ActiveMode;

  prtyFloat Window{ "Window", 0.25f };
  prtyFloat Level{ "Level", 0.5f };
  prtyFloat Isovalue{ "Isovalue", 0.5f };
  prtyFloat Isorange{ "Isorange", 0.1f };
  prtyFloat PctLow{ "PctLow", 0.5f };
  prtyFloat PctHigh{ "PctHigh", 0.98f };
  prtyUint16 MaxU16{ "MaxU16", 65535 };
  prtyUint16 MinU16{ "MinU16", 0 };
  prtyControlPointVector CustomControlPoints{ "CustomControlPoints",
                                              std::vector<LutControlPoint>{ { 0.0f, 0.0f }, { 1.0f, 1.0f } } };
};

class ChannelSettingsDataObject
{
public:
  ChannelSettingsDataObject() {}

  prtyColor DiffuseColor{ "DiffuseColor", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f) };
  prtyColor SpecularColor{ "SpecularColor", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f) };
  prtyColor EmissiveColor{ "EmissiveColor", glm::vec4(0.0f, 0.0f, 0.0f, 1.0f) };
  prtyFloat Roughness{ "Roughness", 0.5f };
  prtyFloat Opacity{ "Opacity", 1.0f };
  prtyBoolean Enabled{ "Enabled", true };
  prtyFloat Labels{ "Labels", 0.0f };

  // a colormap has a name and a set of color control points
  //  std::string name = "none";
  // std::vector<ControlPointSettings_V1> stops;
  // where a ControlPointSettings_V1 is  std::pair<float, std::array<float,4>>

  prtyText ColormapName{ "ColormapName", ColorRamp::NO_COLORMAP_NAME };
  prtyColorControlPointVector Colormap{ "Colormap",
                                        std::vector<ColorControlPoint>{ { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
                                                                        { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f } } };

  GradientDataObject GradientData;
};
