#include "AppearanceDataObject.hpp"

#include "Enumerations.h"
#include "Logging.h"

AppearanceDataOject::AppearanceDataOject(RenderSettings* renderSettings)
  : m_renderSettings(renderSettings)
{
  updatePropsFromRenderSettings();
  // hook up properties to update the underlying camera
  // RendererType.addCallback([this](prtyProperty<int>* p, bool fromUi) {
  //   if (fromUi)
  //     update();
  // });
  // ShadingType.addCallback([this](prtyProperty<int>* p, bool fromUi) {
  //   if (fromUi)
  //     update();
  // });
  // DensityScale.addCallback([this](prtyProperty<float>* p, bool fromUi) {
  //   if (fromUi)
  //     update();
  // });
  // GradientFactor.addCallback([this](prtyProperty<float>* p, bool fromUi) {
  //   if (fromUi)
  //     update();
  // });
  // StepSizePrimaryRay.addCallback([this](prtyProperty<float>* p, bool fromUi) {
  //   if (fromUi)
  //     update();
  // });
  // StepSizeSecondaryRay.addCallback([this](prtyProperty<float>* p, bool fromUi) {
  //   if (fromUi)
  //     update();
  // });
  // Interpolate.addCallback([this](prtyProperty<bool>* p, bool fromUi) {
  //   if (fromUi)
  //     update();
  // });
  // BackgroundColor.addCallback([this](prtyProperty<glm::vec3>* p, bool fromUi) {
  //   if (fromUi)
  //     update();
  // });
  // ShowBoundingBox.addCallback([this](prtyProperty<bool>* p, bool fromUi) {
  //   if (fromUi)
  //     update();
  // });
  // BoundingBoxColor.addCallback([this](prtyProperty<glm::vec3>* p, bool fromUi) {
  //   if (fromUi)
  //     update();
  // });
  // ShowScaleBar.addCallback([this](prtyProperty<bool>* p, bool fromUi) {
  //   if (fromUi)
  //     update();
  // });
}

void
AppearanceDataOject::updatePropsFromRenderSettings()
{
  if (m_renderSettings) {
    ShadingType.SetValue(m_renderSettings->m_RenderSettings.m_ShadingType);
    RendererType.SetValue(m_renderSettings->m_rendererType);
    DensityScale.SetValue(m_renderSettings->m_RenderSettings.m_DensityScale);
    GradientFactor.SetValue(m_renderSettings->m_RenderSettings.m_GradientFactor);
    StepSizePrimaryRay.SetValue(m_renderSettings->m_RenderSettings.m_StepSizeFactor);
    StepSizeSecondaryRay.SetValue(m_renderSettings->m_RenderSettings.m_StepSizeFactorShadow);
    Interpolate.SetValue(m_renderSettings->m_RenderSettings.m_InterpolatedVolumeSampling);
  }
  if (m_scene) {
    BackgroundColor.SetValue(glm::vec3(m_scene->m_material.m_backgroundColor[0],
                                       m_scene->m_material.m_backgroundColor[1],
                                       m_scene->m_material.m_backgroundColor[2]));
    ShowBoundingBox.SetValue(m_scene->m_material.m_showBoundingBox);
    BoundingBoxColor.SetValue(glm::vec3(m_scene->m_material.m_boundingBoxColor[0],
                                        m_scene->m_material.m_boundingBoxColor[1],
                                        m_scene->m_material.m_boundingBoxColor[2]));
    ShowScaleBar.SetValue(m_scene->m_showScaleBar);
  }
}
void
AppearanceDataOject::update()
{
  // update low-level object from properties
  if (m_renderSettings) {
    m_renderSettings->m_RenderSettings.m_ShadingType = ShadingType.GetValue();
    m_renderSettings->m_rendererType = RendererType.GetValue();
    m_renderSettings->m_RenderSettings.m_DensityScale = DensityScale.GetValue();
    m_renderSettings->m_RenderSettings.m_GradientFactor = GradientFactor.GetValue();
    m_renderSettings->m_RenderSettings.m_StepSizeFactor = StepSizePrimaryRay.GetValue();
    m_renderSettings->m_RenderSettings.m_StepSizeFactorShadow = StepSizeSecondaryRay.GetValue();
    m_renderSettings->m_RenderSettings.m_InterpolatedVolumeSampling = Interpolate.GetValue();
    m_scene->m_material.m_backgroundColor[0] = BackgroundColor.GetValue().x;
    m_scene->m_material.m_backgroundColor[1] = BackgroundColor.GetValue().y;
    m_scene->m_material.m_backgroundColor[2] = BackgroundColor.GetValue().z;
    m_scene->m_material.m_showBoundingBox = ShowBoundingBox.GetValue();
    m_scene->m_material.m_boundingBoxColor[0] = BoundingBoxColor.GetValue().x;
    m_scene->m_material.m_boundingBoxColor[1] = BoundingBoxColor.GetValue().y;
    m_scene->m_material.m_boundingBoxColor[2] = BoundingBoxColor.GetValue().z;
    m_scene->m_showScaleBar = ShowScaleBar.GetValue();

    m_renderSettings->m_DirtyFlags.SetFlag(RenderParamsDirty);
    m_renderSettings->m_DirtyFlags.SetFlag(TransferFunctionDirty);
    m_renderSettings->m_DirtyFlags.SetFlag(LightsDirty);
  }
}