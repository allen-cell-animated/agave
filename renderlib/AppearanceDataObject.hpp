#pragma once

#include "core/prty/prtyProperty.h"
#include "RenderSettings.h"
#include "glm.h"

class AppearanceDataObject
{
public:
  AppearanceDataObject(RenderSettings* rs);

  prtyProperty<int> RendererType{ "RendererType", 0 };
  prtyProperty<int> ShadingType{ "ShadingType", 0 };
  prtyProperty<float> DensityScale{ "DensityScale", 1.0f };
  prtyProperty<float> GradientFactor{ "GradientFactor", 0.5f };
  prtyProperty<float> StepSizePrimaryRay{ "StepSizePrimaryRay", 1.0f };
  prtyProperty<float> StepSizeSecondaryRay{ "StepSizeSecondaryRay", 1.0f };
  prtyProperty<bool> Interpolate{ "Interpolate", false };
  prtyProperty<glm::vec3> BackgroundColor{ "BackgroundColor", glm::vec3(0.0f, 0.0f, 0.0f) };
  prtyProperty<bool> ShowBoundingBox{ "ShowBoundingBox", false };
  prtyProperty<glm::vec3> BoundingBoxColor{ "BoundingBoxColor", glm::vec3(1.0f, 1.0f, 1.0f) };
  prtyProperty<bool> ShowScaleBar{ "ShowScaleBar", false };

  RenderSettings* m_renderSettings;

  void updatePropsFromRenderSettings();

private:
  void update();
};
