#include "AppearanceDataObject.hpp"

#include "Enumerations.h"
#include "Logging.h"

AppearanceDataObject::AppearanceDataObject(RenderSettings* renderSettings)
  : m_renderSettings(renderSettings)
{
  updatePropsFromRenderSettings();
  // hook up properties to update the underlying camera
  RendererType.addCallback([this](prtyProperty<int>* p, bool) { update(); });
  ShadingType.addCallback([this](prtyProperty<int>* p, bool) { update(); });
  DensityScale.addCallback([this](prtyProperty<float>* p, bool) { update(); });
  GradientFactor.addCallback([this](prtyProperty<float>* p, bool) { update(); });
  StepSizePrimaryRay.addCallback([this](prtyProperty<float>* p, bool) { update(); });
  StepSizeSecondaryRay.addCallback([this](prtyProperty<float>* p, bool) { update(); });
  Interpolate.addCallback([this](prtyProperty<bool>* p, bool) { update(); });
  BackgroundColor.addCallback([this](prtyProperty<glm::vec3>* p, bool) { update(); });
  ShowBoundingBox.addCallback([this](prtyProperty<bool>* p, bool) { update(); });
  BoundingBoxColor.addCallback([this](prtyProperty<glm::vec3>* p, bool) { update(); });
  ShowScaleBar.addCallback([this](prtyProperty<bool>* p, bool) { update(); });
}

void
AppearanceDataObject::updatePropsFromRenderSettings()
{
  if (m_renderSettings) {
    ShadingType.set(m_renderSettings->m_RenderSettings.m_ShadingType);
    // RendererType.set(m_renderSettings->m_RenderSettings.m_RendererType);
    DensityScale.set(m_renderSettings->m_RenderSettings.m_DensityScale);
    GradientFactor.set(m_renderSettings->m_RenderSettings.m_GradientFactor);
    StepSizePrimaryRay.set(m_renderSettings->m_RenderSettings.m_StepSizeFactor);
    StepSizeSecondaryRay.set(m_renderSettings->m_RenderSettings.m_StepSizeFactorShadow);
    Interpolate.set(m_renderSettings->m_RenderSettings.m_InterpolatedVolumeSampling);
    // BackgroundColor.set(glm::vec3(m_renderSettings->m_RenderSettings.m_BackgroundColor[0],
    //                               m_renderSettings->m_RenderSettings.m_BackgroundColor[1],
    //                               m_renderSettings->m_RenderSettings.m_BackgroundColor[2]));
    // ShowBoundingBox.set(m_renderSettings->m_RenderSettings.m_ShowBoundingBox);
    // BoundingBoxColor.set(glm::vec3(m_renderSettings->m_RenderSettings.m_BoundingBoxColor[0],
    //                                m_renderSettings->m_RenderSettings.m_BoundingBoxColor[1],
    //                                m_renderSettings->m_RenderSettings.m_BoundingBoxColor[2]));
    // ShowScaleBar.set(m_renderSettings->m_RenderSettings.m_ShowScaleBar);
  }
}
void
AppearanceDataObject::update()
{
  // update low-level object from properties
  if (m_renderSettings) {
    m_renderSettings->m_RenderSettings.m_ShadingType = ShadingType.get();
    // m_renderSettings->m_RenderSettings.m_RendererType = RendererType.get();
    m_renderSettings->m_RenderSettings.m_DensityScale = DensityScale.get();
    m_renderSettings->m_RenderSettings.m_GradientFactor = GradientFactor.get();
    m_renderSettings->m_RenderSettings.m_StepSizeFactor = StepSizePrimaryRay.get();
    m_renderSettings->m_RenderSettings.m_StepSizeFactorShadow = StepSizeSecondaryRay.get();
    m_renderSettings->m_RenderSettings.m_InterpolatedVolumeSampling = Interpolate.get();
    // m_renderSettings->m_RenderSettings.m_BackgroundColor[0] = BackgroundColor.get().x;
    // m_renderSettings->m_RenderSettings.m_BackgroundColor[1] = BackgroundColor.get().y;
    // m_renderSettings->m_RenderSettings.m_BackgroundColor[2] = BackgroundColor.get().z;
    // m_renderSettings->m_RenderSettings.m_ShowBoundingBox = ShowBoundingBox.get();
    // m_renderSettings->m_RenderSettings.m_BoundingBoxColor[0] = BoundingBoxColor.get().x;
    // m_renderSettings->m_RenderSettings.m_BoundingBoxColor[1] = BoundingBoxColor.get().y;
    // m_renderSettings->m_RenderSettings.m_BoundingBoxColor[2] = BoundingBoxColor.get().z;
    // m_renderSettings->m_RenderSettings.m_ShowScaleBar = ShowScaleBar.get();

    m_renderSettings->m_DirtyFlags.SetFlag(RenderParamsDirty);
    m_renderSettings->m_DirtyFlags.SetFlag(TransferFunctionDirty);
    m_renderSettings->m_DirtyFlags.SetFlag(LightsDirty);
  }
}