#include "AppearanceWidget.h"
#include "RenderSettings.h"

#include "qtControls/controlFactory.h"

#include "renderlib/uiInfo.hpp"
#include "renderlib/AppearanceObject.hpp"
#include "renderlib/ViewerWindow.h"

#include <QLabel>

QAppearanceWidget2::QAppearanceWidget2(QWidget* pParent, RenderSettings* rs, ViewerWindow* vw, AppearanceObject* cdo)
  : QWidget(pParent)
  , m_MainLayout()
  , m_renderSettings(rs)
  , m_appearanceObject(cdo)
{
  Controls::initFormLayout(m_MainLayout);
  setLayout(&m_MainLayout);
  if (m_appearanceObject) {
    createFlatList(&m_MainLayout, m_appearanceObject);
  }
  // // loop over all properties in cameraobject. for each property, add a callback that updates the rendersetttings
  // // cameradirty flags
  // for (const auto& prop : m_cameraObject->GetList()) {
  //   if (prop) {
  //     prop->GetProperty(0)->AddCallback(new prtyCallbackLambda([this](prtyProperty* i_Property, bool i_bDirty) {
  //       if (i_bDirty) {
  //         m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
  //       }
  //     }));
  //   }
  // }

  QComboBox* rendererType = addRow(*m_appearanceDataObject->getRendererTypeUiInfo());
  m_MainLayout.addRow("Renderer", rendererType);
  QComboBox* shadingType = addRow(*m_appearanceDataObject->getShadingTypeUiInfo());
  m_MainLayout.addRow("Shading Type", shadingType);
  QNumericSlider* densityScale = addRow(*m_appearanceDataObject->getDensityScaleUiInfo());
  m_MainLayout.addRow("Scattering Density", densityScale);
  QNumericSlider* gradientFactor = addRow(*m_appearanceDataObject->getGradientFactorUiInfo());
  m_MainLayout.addRow("Shading Type Mixture", gradientFactor);
  QNumericSlider* stepSizePrimaryRay = addRow(*m_appearanceDataObject->getStepSizePrimaryRayUiInfo());
  m_MainLayout.addRow("Step Size Primary Ray", stepSizePrimaryRay);
  QNumericSlider* stepSizeSecondaryRay = addRow(*m_appearanceDataObject->getStepSizeSecondaryRayUiInfo());
  m_MainLayout.addRow("Step Size Secondary Ray", stepSizeSecondaryRay);
  QCheckBox* interpolateCheckBox = addRow(*m_appearanceDataObject->getInterpolateUiInfo());
  m_MainLayout.addRow("Interpolate", interpolateCheckBox);
  QColorPushButton* backgroundColorButton = addRow(*m_appearanceDataObject->getBackgroundColorUiInfo());
  m_MainLayout.addRow("Background Color", backgroundColorButton);
  QCheckBox* showBoundingBoxCheckBox = addRow(*m_appearanceDataObject->getShowBoundingBoxUiInfo());
  m_MainLayout.addRow("Show Bounding Box", showBoundingBoxCheckBox);
  QColorPushButton* boundingBoxColorButton = addRow(*m_appearanceDataObject->getBoundingBoxColorUiInfo());
  m_MainLayout.addRow("Bounding Box Color", boundingBoxColorButton);
  QCheckBox* showScaleBarCheckBox = addRow(*m_appearanceDataObject->getShowScaleBarUiInfo());
  m_MainLayout.addRow("Show Scale Bar", showScaleBarCheckBox);

  QObject::connect(rendererType, &QComboBox::currentIndexChanged, [this, vw](int index) { vw->setRenderer(index); });
  QObject::connect(shadingType, &QComboBox::currentIndexChanged, [this, gradientFactor](int index) {
    gradientFactor->setEnabled(index == 2);
  });
}

QSize
QAppearanceWidget2::sizeHint() const
{
  return QSize(20, 20);
}
