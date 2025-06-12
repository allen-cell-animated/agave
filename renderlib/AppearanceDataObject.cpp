#include "AppearanceDataObject.hpp"

#include "Enumerations.h"
#include "Logging.h"

AppearanceDataObject::AppearanceDataObject(RenderSettings* renderSettings, Scene* scene)
  : m_renderSettings(renderSettings)
  , m_scene(scene)
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
    RendererType.set(m_renderSettings->m_rendererType);
    DensityScale.set(m_renderSettings->m_RenderSettings.m_DensityScale);
    GradientFactor.set(m_renderSettings->m_RenderSettings.m_GradientFactor);
    StepSizePrimaryRay.set(m_renderSettings->m_RenderSettings.m_StepSizeFactor);
    StepSizeSecondaryRay.set(m_renderSettings->m_RenderSettings.m_StepSizeFactorShadow);
    Interpolate.set(m_renderSettings->m_RenderSettings.m_InterpolatedVolumeSampling);
  }
  if (m_scene) {
    BackgroundColor.set(glm::vec3(m_scene->m_material.m_backgroundColor[0],
                                  m_scene->m_material.m_backgroundColor[1],
                                  m_scene->m_material.m_backgroundColor[2]));
    ShowBoundingBox.set(m_scene->m_material.m_showBoundingBox);
    BoundingBoxColor.set(glm::vec3(m_scene->m_material.m_boundingBoxColor[0],
                                   m_scene->m_material.m_boundingBoxColor[1],
                                   m_scene->m_material.m_boundingBoxColor[2]));
    ShowScaleBar.set(m_scene->m_showScaleBar);
  }
}
void
AppearanceDataObject::update()
{
  // update low-level object from properties
  if (m_renderSettings) {
    m_renderSettings->m_RenderSettings.m_ShadingType = ShadingType.get();
    m_renderSettings->m_rendererType = RendererType.get();
    m_renderSettings->m_RenderSettings.m_DensityScale = DensityScale.get();
    m_renderSettings->m_RenderSettings.m_GradientFactor = GradientFactor.get();
    m_renderSettings->m_RenderSettings.m_StepSizeFactor = StepSizePrimaryRay.get();
    m_renderSettings->m_RenderSettings.m_StepSizeFactorShadow = StepSizeSecondaryRay.get();
    m_renderSettings->m_RenderSettings.m_InterpolatedVolumeSampling = Interpolate.get();
    m_scene->m_material.m_backgroundColor[0] = BackgroundColor.get().x;
    m_scene->m_material.m_backgroundColor[1] = BackgroundColor.get().y;
    m_scene->m_material.m_backgroundColor[2] = BackgroundColor.get().z;
    m_scene->m_material.m_showBoundingBox = ShowBoundingBox.get();
    m_scene->m_material.m_boundingBoxColor[0] = BoundingBoxColor.get().x;
    m_scene->m_material.m_boundingBoxColor[1] = BoundingBoxColor.get().y;
    m_scene->m_material.m_boundingBoxColor[2] = BoundingBoxColor.get().z;
    m_scene->m_showScaleBar = ShowScaleBar.get();

    m_renderSettings->m_DirtyFlags.SetFlag(RenderParamsDirty);
    m_renderSettings->m_DirtyFlags.SetFlag(TransferFunctionDirty);
    m_renderSettings->m_DirtyFlags.SetFlag(LightsDirty);
  }
}