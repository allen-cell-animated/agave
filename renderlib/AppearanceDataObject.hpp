#pragma once

#include "core/prty/prtyInt8.hpp"
#include "core/prty/prtyFloat.hpp"
#include "core/prty/prtyBoolean.hpp"
#include "AppScene.h"
#include "RenderSettings.h"
#include "glm.h"

class AppearanceDataObject
{
public:
  AppearanceDataObject(RenderSettings* rs, Scene* scene);

  prtyInt8 RendererType{ "RendererType", 0 };
  prtyInt8 ShadingType{ "ShadingType", 0 };
  prtyFloat DensityScale{ "DensityScale", 1.0f };
  prtyFloat GradientFactor{ "GradientFactor", 0.5f };
  prtyFloat StepSizePrimaryRay{ "StepSizePrimaryRay", 1.0f };
  prtyFloat StepSizeSecondaryRay{ "StepSizeSecondaryRay", 1.0f };
  prtyBoolean Interpolate{ "Interpolate", false };
  prtyVec3 BackgroundColor{ "BackgroundColor", glm::vec3(0.0f, 0.0f, 0.0f) };
  prtyBoolean ShowBoundingBox{ "ShowBoundingBox", false };
  prtyVec3 BoundingBoxColor{ "BoundingBoxColor", glm::vec3(1.0f, 1.0f, 1.0f) };
  prtyBoolean ShowScaleBar{ "ShowScaleBar", false };

  RenderSettings* m_renderSettings;
  Scene* m_scene;

  void updatePropsFromRenderSettings();

private:
  void update();
};
