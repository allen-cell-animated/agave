#pragma once

#include "GradientData.h"
#include "core/prty/prtyColor.hpp"
#include "core/prty/prtyFloat.hpp"
#include "core/prty/prtyEnum.hpp"
#include "core/prty/prtyBoolean.hpp"
#include "core/prty/prtyVector3d.hpp"

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

  prtyColorMap Colormap{ "Colormap", ColorRamp() };

  GradientDataObject GradientData;
};
