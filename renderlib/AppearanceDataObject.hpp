#pragma once

#include "core/prty/prtyIntegerTemplate.hpp"
#include "core/prty/prtyEnum.hpp"
#include "core/prty/prtyFloat.hpp"
#include "core/prty/prtyBoolean.hpp"
#include "core/prty/prtyVector3d.hpp"
#include "core/prty/prtyColor.hpp"
#include "glm.h"

class AppearanceDataObject
{
public:
  AppearanceDataObject()
  {
    RendererType.SetEnumTag(0, "Ray march blending");
    RendererType.SetEnumTag(1, "Path traced");

    ShadingType.SetEnumTag(0, "BRDF Only");
    ShadingType.SetEnumTag(1, "Phase Function Only");
    ShadingType.SetEnumTag(2, "Mixed");
  }

  prtyEnum RendererType{ "RendererType", 0 };
  prtyEnum ShadingType{ "ShadingType", 0 };
  prtyFloat DensityScale{ "DensityScale", 1.0f };
  prtyFloat GradientFactor{ "GradientFactor", 0.5f };
  prtyFloat StepSizePrimaryRay{ "StepSizePrimaryRay", 1.0f };
  prtyFloat StepSizeSecondaryRay{ "StepSizeSecondaryRay", 1.0f };
  prtyBoolean Interpolate{ "Interpolate", false };
  prtyColor BackgroundColor{ "BackgroundColor", glm::vec4(0.0f, 0.0f, 0.0f, 1.0f) };
  prtyBoolean ShowBoundingBox{ "ShowBoundingBox", false };
  prtyColor BoundingBoxColor{ "BoundingBoxColor", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f) };
  prtyBoolean ShowScaleBar{ "ShowScaleBar", false };
};
