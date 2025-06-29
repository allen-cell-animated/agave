#pragma once

#include "core/prty/prtyInt8.hpp"
#include "core/prty/prtyFloat.hpp"
#include "core/prty/prtyBoolean.hpp"
#include "core/prty/prtyVector3d.hpp"
#include "core/prty/prtyColor.hpp"
#include "glm.h"

class AppearanceDataObject
{
public:
  AppearanceDataObject();

  prtyInt8 RendererType{ "RendererType", 1 };
  prtyInt8 ShadingType{ "ShadingType", 0 };
  prtyFloat DensityScale{ "DensityScale", 1.0f };
  prtyFloat GradientFactor{ "GradientFactor", 0.5f };
  prtyFloat StepSizePrimaryRay{ "StepSizePrimaryRay", 4.0f };
  prtyFloat StepSizeSecondaryRay{ "StepSizeSecondaryRay", 4.0f };
  prtyBoolean Interpolate{ "Interpolate", true };
  prtyColor BackgroundColor{ "BackgroundColor", glm::vec4(0.0f, 0.0f, 0.0f, 1.0f) };
  prtyBoolean ShowBoundingBox{ "ShowBoundingBox", true };
  prtyColor BoundingBoxColor{ "BoundingBoxColor", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f) };
  prtyBoolean ShowScaleBar{ "ShowScaleBar", false };
};
