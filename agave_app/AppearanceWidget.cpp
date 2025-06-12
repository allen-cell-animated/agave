#include "AppearanceWidget.h"
#include "RenderSettings.h"

#include "qtControls/controlFactory.h"

#include "renderlib/uiInfo.hpp"
#include "renderlib/AppearanceUiDescription.hpp"
#include "renderlib/ViewerWindow.h"

#include <QLabel>

QAppearanceWidget2::QAppearanceWidget2(QWidget* pParent,
                                       RenderSettings* rs,
                                       ViewerWindow* vw,
                                       AppearanceDataObject* cdo)
  : QWidget(pParent)
  , m_MainLayout()
  , m_renderSettings(rs)
  , m_appearanceDataObject(cdo)
{
  Controls::initFormLayout(m_MainLayout);
  setLayout(&m_MainLayout);

  QComboBox* rendererType = addRow(AppearanceUiDescription::m_rendererType, &m_appearanceDataObject->RendererType);
  m_MainLayout.addRow("Renderer", rendererType);
  QComboBox* shadingType = addRow(AppearanceUiDescription::m_shadingType, &m_appearanceDataObject->ShadingType);
  m_MainLayout.addRow("Shading Type", shadingType);
  QNumericSlider* densityScale = addRow(AppearanceUiDescription::m_densityScale, &m_appearanceDataObject->DensityScale);
  m_MainLayout.addRow("Scattering Density", densityScale);
  QNumericSlider* gradientFactor =
    addRow(AppearanceUiDescription::m_gradientFactor, &m_appearanceDataObject->GradientFactor);
  m_MainLayout.addRow("Shading Type Mixture", gradientFactor);
  QNumericSlider* stepSizePrimaryRay =
    addRow(AppearanceUiDescription::m_stepSizePrimaryRay, &m_appearanceDataObject->StepSizePrimaryRay);
  m_MainLayout.addRow("Step Size Primary Ray", stepSizePrimaryRay);
  QNumericSlider* stepSizeSecondaryRay =
    addRow(AppearanceUiDescription::m_stepSizeSecondaryRay, &m_appearanceDataObject->StepSizeSecondaryRay);
  m_MainLayout.addRow("Step Size Secondary Ray", stepSizeSecondaryRay);
  QCheckBox* interpolateCheckBox = addRow(AppearanceUiDescription::m_interpolate, &m_appearanceDataObject->Interpolate);
  m_MainLayout.addRow("Interpolate", interpolateCheckBox);
  QColorPushButton* backgroundColorButton =
    addRow(AppearanceUiDescription::m_backgroundColor, &m_appearanceDataObject->BackgroundColor);
  m_MainLayout.addRow("Background Color", backgroundColorButton);
  QCheckBox* showBoundingBoxCheckBox =
    addRow(AppearanceUiDescription::m_showBoundingBox, &m_appearanceDataObject->ShowBoundingBox);
  m_MainLayout.addRow("Show Bounding Box", showBoundingBoxCheckBox);
  QColorPushButton* boundingBoxColorButton =
    addRow(AppearanceUiDescription::m_boundingBoxColor, &m_appearanceDataObject->BoundingBoxColor);
  m_MainLayout.addRow("Bounding Box Color", boundingBoxColorButton);
  QCheckBox* showScaleBarCheckBox =
    addRow(AppearanceUiDescription::m_showScaleBar, &m_appearanceDataObject->ShowScaleBar);
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
