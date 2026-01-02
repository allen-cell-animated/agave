#pragma once

#include "core/prty/prtyFloat.hpp"
#include "core/prty/prtyEnum.hpp"
#include "core/prty/prtyBoolean.hpp"
#include "core/prty/prtyVector3d.hpp"

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
  prty std::vector<LutControlPoint> m_customControlPoints = { { 0.0f, 0.0f }, { 1.0f, 1.0f } };
};

class ChannelSettingsDataObject
{
public:
  ChannelSettingsDataObject()
  {
    ExposureIterations.SetEnumTag(0, "1");
    ExposureIterations.SetEnumTag(1, "2");
    ExposureIterations.SetEnumTag(2, "4");
    ExposureIterations.SetEnumTag(3, "8");

    ProjectionMode.SetEnumTag(0, "Perspective");
    ProjectionMode.SetEnumTag(1, "Orthographic");
  }

  prtyFloat Exposure{ "Exposure", 0.75f };
  prtyEnum ExposureIterations{ "ExposureIterations", 0 };
  prtyBoolean NoiseReduction{ "NoiseReduction", false };
  prtyFloat ApertureSize{ "ApertureSize", 0.0f };
  prtyFloat FieldOfView{ "FieldOfView", 30.0f }; // degrees
  prtyFloat FocalDistance{ "FocalDistance", 0.0f };

  prtyVector3d Position{ "Position", glm::vec3(0.0f, 0.0f, 0.0f) };
  prtyVector3d Target{ "Target", glm::vec3(0.0f, 0.0f, -1.0f) };
  prtyFloat NearPlane{ "NearPlane", 0.1f };
  prtyFloat FarPlane{ "FarPlane", 1000.0f };
  prtyFloat Roll{ "Roll", 0.0f }; // tilt angle in degrees

  prtyFloat OrthoScale{ "OrthoScale", 1.0f };     // orthographic scale for orthographic projection
  prtyEnum ProjectionMode{ "ProjectionMode", 0 }; // 0 = perspective, 1 = orthographic

  prtyColor DiffuseColor{ "DiffuseColor", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f) };
  prtyColor SpecularColor{ "SpecularColor", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f) };
  prtyColor EmissiveColor{ "EmissiveColor", glm::vec4(0.0f, 0.0f, 0.0f, 1.0f) };
  prtyFloat Roughness{ "Roughness", 0.5f };
  prtyFloat Opacity{ "Opacity", 1.0f };
  prtyBoolean Enabled{ "Enabled", true };
  prtyFloat Labels{ "Labels", 0.0f };

  prtyColorMap Colormap{ "Colormap", ColorRamp() };

  GradientData m_gradientData[MAX_CPU_CHANNELS];
};
