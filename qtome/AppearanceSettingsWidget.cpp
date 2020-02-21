#include "AppearanceSettingsWidget.h"
#include "QRenderSettings.h"
#include "RangeWidget.h"
#include "Section.h"

#include "ImageXYZC.h"
#include "renderlib/AppScene.h"
#include "renderlib/FileReader.h"
#include "renderlib/Logging.h"
#include "renderlib/RenderSettings.h"
#include "tfeditor/gradients.h"

#include <QApplication>
#include <QFormLayout>
#include <QLinearGradient>

QAppearanceSettingsWidget::QAppearanceSettingsWidget(QWidget* pParent, QRenderSettings* qrs, RenderSettings* rs)
  : QGroupBox(pParent)
  , m_MainLayout()
  , m_DensityScaleSlider()
  , m_RendererType()
  , m_ShadingType()
  , m_GradientFactorSlider()
  , m_StepSizePrimaryRaySlider()
  , m_StepSizeSecondaryRaySlider()
  , m_qrendersettings(qrs)
  , m_scene(nullptr)
{
  Controls::initFormLayout(m_MainLayout);
  setLayout(&m_MainLayout);

  m_RendererType.addItem("OpenGL simple", 0);
  m_RendererType.addItem("Path Traced", 1);
  // m_RendererType.addItem("OpenGL full", 2);
  m_RendererType.setCurrentIndex(1);
  m_MainLayout.addRow("Renderer", &m_RendererType);

  m_DensityScaleSlider.setRange(0.001, 100.0);
  m_DensityScaleSlider.setDecimals(3);
  m_DensityScaleSlider.setValue(rs->m_RenderSettings.m_DensityScale);
  m_MainLayout.addRow("Scattering Density", &m_DensityScaleSlider);

  m_ShadingType.addItem("BRDF Only", 0);
  m_ShadingType.addItem("Phase Function Only", 1);
  m_ShadingType.addItem("Mixed", 2);
  m_ShadingType.setCurrentIndex(rs->m_RenderSettings.m_ShadingType);
  m_MainLayout.addRow("Shading Type", &m_ShadingType);

  m_GradientFactorSlider.setRange(0.001, 100.0);
  m_GradientFactorSlider.setDecimals(3);
  m_GradientFactorSlider.setValue(rs->m_RenderSettings.m_GradientFactor);
  m_MainLayout.addRow("Shading Type Mixture", &m_GradientFactorSlider);

  QObject::connect(&m_DensityScaleSlider, SIGNAL(valueChanged(double)), this, SLOT(OnSetDensityScale(double)));
  QObject::connect(&m_GradientFactorSlider, SIGNAL(valueChanged(double)), this, SLOT(OnSetGradientFactor(double)));

  m_StepSizePrimaryRaySlider.setRange(0.1, 100.0);
  m_StepSizePrimaryRaySlider.setValue(rs->m_RenderSettings.m_StepSizeFactor);
  m_StepSizePrimaryRaySlider.setDecimals(3);
  m_MainLayout.addRow("Primary Ray Step Size", &m_StepSizePrimaryRaySlider);

  QObject::connect(
    &m_StepSizePrimaryRaySlider, SIGNAL(valueChanged(double)), this, SLOT(OnSetStepSizePrimaryRay(double)));

  m_StepSizeSecondaryRaySlider.setRange(0.1, 100.0);
  m_StepSizeSecondaryRaySlider.setValue(rs->m_RenderSettings.m_StepSizeFactorShadow);
  m_StepSizeSecondaryRaySlider.setDecimals(3);
  m_MainLayout.addRow("Secondary Ray Step Size", &m_StepSizeSecondaryRaySlider);

  QObject::connect(
    &m_StepSizeSecondaryRaySlider, SIGNAL(valueChanged(double)), this, SLOT(OnSetStepSizeSecondaryRay(double)));

  m_backgroundColorButton.SetColor(QColor(0, 0, 0, 0), true);
  m_MainLayout.addRow("BackgroundColor", &m_backgroundColorButton);

  QObject::connect(&m_backgroundColorButton, &QColorPushButton::currentColorChanged, [this](const QColor& c) {
    this->OnBackgroundColorChanged(c);
  });

  m_scaleSection = new Section("Volume Scale", 0);
  auto* scaleSectionLayout = new QGridLayout();
  scaleSectionLayout->addWidget(new QLabel("X"), 0, 0);
  m_xscaleSpinner = new QDoubleSpinner();
  m_xscaleSpinner->setValue(1.0);
  scaleSectionLayout->addWidget(m_xscaleSpinner, 0, 1);
  QObject::connect(m_xscaleSpinner,
                   QOverload<double>::of(&QDoubleSpinner::valueChanged),
                   this,
                   &QAppearanceSettingsWidget::OnSetScaleX);
  scaleSectionLayout->addWidget(new QLabel("Y"), 1, 0);
  m_yscaleSpinner = new QDoubleSpinner();
  m_yscaleSpinner->setValue(1.0);
  scaleSectionLayout->addWidget(m_yscaleSpinner, 1, 1);
  QObject::connect(m_yscaleSpinner,
                   QOverload<double>::of(&QDoubleSpinner::valueChanged),
                   this,
                   &QAppearanceSettingsWidget::OnSetScaleY);
  scaleSectionLayout->addWidget(new QLabel("Z"), 2, 0);
  m_zscaleSpinner = new QDoubleSpinner();
  m_zscaleSpinner->setValue(1.0);
  scaleSectionLayout->addWidget(m_zscaleSpinner, 2, 1);
  QObject::connect(m_zscaleSpinner,
                   QOverload<double>::of(&QDoubleSpinner::valueChanged),
                   this,
                   &QAppearanceSettingsWidget::OnSetScaleZ);

  m_scaleSection->setContentLayout(*scaleSectionLayout);
  m_MainLayout.addRow(m_scaleSection);

  m_clipRoiSection = new Section("ROI", 0);
  auto* roiSectionLayout = new QGridLayout();
  roiSectionLayout->addWidget(new QLabel("X"), 0, 0);
  m_roiX = new RangeWidget(Qt::Horizontal);
  m_roiX->setRange(0, 100);
  m_roiX->setFirstValue(0);
  m_roiX->setSecondValue(100);
  roiSectionLayout->addWidget(m_roiX, 0, 1);
  QObject::connect(m_roiX, &RangeWidget::firstValueChanged, this, &QAppearanceSettingsWidget::OnSetRoiXMin);
  QObject::connect(m_roiX, &RangeWidget::secondValueChanged, this, &QAppearanceSettingsWidget::OnSetRoiXMax);
  roiSectionLayout->addWidget(new QLabel("Y"), 1, 0);
  m_roiY = new RangeWidget(Qt::Horizontal);
  m_roiY->setRange(0, 100);
  m_roiY->setFirstValue(0);
  m_roiY->setSecondValue(100);
  roiSectionLayout->addWidget(m_roiY, 1, 1);
  QObject::connect(m_roiY, &RangeWidget::firstValueChanged, this, &QAppearanceSettingsWidget::OnSetRoiYMin);
  QObject::connect(m_roiY, &RangeWidget::secondValueChanged, this, &QAppearanceSettingsWidget::OnSetRoiYMax);
  roiSectionLayout->addWidget(new QLabel("Z"), 2, 0);
  m_roiZ = new RangeWidget(Qt::Horizontal);
  m_roiZ->setRange(0, 100);
  m_roiZ->setFirstValue(0);
  m_roiZ->setSecondValue(100);
  roiSectionLayout->addWidget(m_roiZ, 2, 1);
  QObject::connect(m_roiZ, &RangeWidget::firstValueChanged, this, &QAppearanceSettingsWidget::OnSetRoiZMin);
  QObject::connect(m_roiZ, &RangeWidget::secondValueChanged, this, &QAppearanceSettingsWidget::OnSetRoiZMax);

  m_clipRoiSection->setContentLayout(*roiSectionLayout);
  m_MainLayout.addRow(m_clipRoiSection);

  Section* section = createLightingControls();
  m_MainLayout.addRow(section);

  QFrame* lineA = new QFrame();
  lineA->setFrameShape(QFrame::HLine);
  lineA->setFrameShadow(QFrame::Sunken);
  m_MainLayout.addRow(lineA);

  QObject::connect(&m_RendererType, SIGNAL(currentIndexChanged(int)), this, SLOT(OnSetRendererType(int)));
  QObject::connect(&m_ShadingType, SIGNAL(currentIndexChanged(int)), this, SLOT(OnSetShadingType(int)));
  // QObject::connect(&gStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));

  QObject::connect(m_qrendersettings, SIGNAL(Changed()), this, SLOT(OnTransferFunctionChanged()));
}

Section*
QAppearanceSettingsWidget::createLightingControls()
{
  Section* section = new Section("Lighting", 0);
  auto* sectionLayout = Controls::createFormLayout();

  m_lt0gui.m_thetaSlider = new QNumericSlider();
  m_lt0gui.m_thetaSlider->setStatusTip("Set angle theta for area light");
  m_lt0gui.m_thetaSlider->setToolTip("Set angle theta for area light");
  m_lt0gui.m_thetaSlider->setRange(0.0, TWO_PI_F);
  m_lt0gui.m_thetaSlider->setSingleStep(TWO_PI_F / 100.0);
  m_lt0gui.m_thetaSlider->setValue(0.0);
  sectionLayout->addRow("AreaLight Theta", m_lt0gui.m_thetaSlider);
  QObject::connect(
    m_lt0gui.m_thetaSlider, &QNumericSlider::valueChanged, this, &QAppearanceSettingsWidget::OnSetAreaLightTheta);

  m_lt0gui.m_phiSlider = new QNumericSlider();
  m_lt0gui.m_phiSlider->setStatusTip("Set angle phi for area light");
  m_lt0gui.m_phiSlider->setToolTip("Set angle phi for area light");
  m_lt0gui.m_phiSlider->setRange(0.0, PI_F);
  m_lt0gui.m_phiSlider->setSingleStep(PI_F / 100.0);
  m_lt0gui.m_phiSlider->setValue(HALF_PI_F);
  sectionLayout->addRow("AreaLight Phi", m_lt0gui.m_phiSlider);
  QObject::connect(
    m_lt0gui.m_phiSlider, &QNumericSlider::valueChanged, this, &QAppearanceSettingsWidget::OnSetAreaLightPhi);

  m_lt0gui.m_sizeSlider = new QNumericSlider();
  m_lt0gui.m_sizeSlider->setStatusTip("Set size for area light");
  m_lt0gui.m_sizeSlider->setToolTip("Set size for area light");
  m_lt0gui.m_sizeSlider->setRange(0.1, 5.0);
  m_lt0gui.m_sizeSlider->setSingleStep(5.0 / 100.0);
  m_lt0gui.m_sizeSlider->setValue(1.0);
  sectionLayout->addRow("AreaLight Size", m_lt0gui.m_sizeSlider);
  QObject::connect(
    m_lt0gui.m_sizeSlider, &QNumericSlider::valueChanged, this, &QAppearanceSettingsWidget::OnSetAreaLightSize);

  m_lt0gui.m_distSlider = new QNumericSlider();
  m_lt0gui.m_distSlider->setStatusTip("Set distance for area light");
  m_lt0gui.m_distSlider->setToolTip("Set distance for area light");
  m_lt0gui.m_distSlider->setRange(0.1, 100.0);
  m_lt0gui.m_distSlider->setSingleStep(1.0);
  m_lt0gui.m_distSlider->setValue(10.0);
  sectionLayout->addRow("AreaLight Distance", m_lt0gui.m_distSlider);
  QObject::connect(
    m_lt0gui.m_distSlider, &QNumericSlider::valueChanged, this, &QAppearanceSettingsWidget::OnSetAreaLightDistance);

  auto* arealightLayout = new QHBoxLayout();
  m_lt0gui.m_intensitySlider = new QNumericSlider();
  m_lt0gui.m_intensitySlider->setStatusTip("Set intensity for area light");
  m_lt0gui.m_intensitySlider->setToolTip("Set intensity for area light");
  m_lt0gui.m_intensitySlider->setRange(0.0, 1000.0);
  m_lt0gui.m_intensitySlider->setSingleStep(1.0);
  m_lt0gui.m_intensitySlider->setValue(100.0);
  arealightLayout->addWidget(m_lt0gui.m_intensitySlider, 1);
  m_lt0gui.m_areaLightColorButton = new QColorPushButton();
  m_lt0gui.m_areaLightColorButton->setStatusTip("Set color for area light");
  m_lt0gui.m_areaLightColorButton->setToolTip("Set color for area light");
  arealightLayout->addWidget(m_lt0gui.m_areaLightColorButton);
  sectionLayout->addRow("AreaLight Intensity", arealightLayout);
  QObject::connect(m_lt0gui.m_areaLightColorButton, &QColorPushButton::currentColorChanged, [this](const QColor& c) {
    this->OnSetAreaLightColor(this->m_lt0gui.m_intensitySlider->value(), c);
  });
  QObject::connect(m_lt0gui.m_intensitySlider, &QNumericSlider::valueChanged, [this](double v) {
    this->OnSetAreaLightColor(v, this->m_lt0gui.m_areaLightColorButton->GetColor());
  });

  auto* skylightTopLayout = new QHBoxLayout();
  m_lt1gui.m_stintensitySlider = new QNumericSlider();
  m_lt1gui.m_stintensitySlider->setRange(0.0, 10.0);
  m_lt1gui.m_stintensitySlider->setValue(1.0);
  skylightTopLayout->addWidget(m_lt1gui.m_stintensitySlider, 1);
  m_lt1gui.m_stColorButton = new QColorPushButton();
  skylightTopLayout->addWidget(m_lt1gui.m_stColorButton);
  sectionLayout->addRow("SkyLight Top", skylightTopLayout);
  QObject::connect(m_lt1gui.m_stColorButton, &QColorPushButton::currentColorChanged, [this](const QColor& c) {
    this->OnSetSkyLightTopColor(this->m_lt1gui.m_stintensitySlider->value(), c);
  });
  QObject::connect(m_lt1gui.m_stintensitySlider, &QNumericSlider::valueChanged, [this](double v) {
    this->OnSetSkyLightTopColor(v, this->m_lt1gui.m_stColorButton->GetColor());
  });

  auto* skylightMidLayout = new QHBoxLayout();
  m_lt1gui.m_smintensitySlider = new QNumericSlider();
  m_lt1gui.m_smintensitySlider->setRange(0.0, 10.0);
  m_lt1gui.m_smintensitySlider->setValue(1.0);
  skylightMidLayout->addWidget(m_lt1gui.m_smintensitySlider, 1);
  m_lt1gui.m_smColorButton = new QColorPushButton();
  skylightMidLayout->addWidget(m_lt1gui.m_smColorButton);
  sectionLayout->addRow("SkyLight Mid", skylightMidLayout);
  QObject::connect(m_lt1gui.m_smColorButton, &QColorPushButton::currentColorChanged, [this](const QColor& c) {
    this->OnSetSkyLightMidColor(this->m_lt1gui.m_smintensitySlider->value(), c);
  });
  QObject::connect(m_lt1gui.m_smintensitySlider, &QNumericSlider::valueChanged, [this](double v) {
    this->OnSetSkyLightMidColor(v, this->m_lt1gui.m_smColorButton->GetColor());
  });

  auto* skylightBotLayout = new QHBoxLayout();
  m_lt1gui.m_sbintensitySlider = new QNumericSlider();
  m_lt1gui.m_sbintensitySlider->setRange(0.0, 10.0);
  m_lt1gui.m_sbintensitySlider->setValue(1.0);
  skylightBotLayout->addWidget(m_lt1gui.m_sbintensitySlider, 1);
  m_lt1gui.m_sbColorButton = new QColorPushButton();
  skylightBotLayout->addWidget(m_lt1gui.m_sbColorButton);
  sectionLayout->addRow("SkyLight Bot", skylightBotLayout);
  QObject::connect(m_lt1gui.m_sbColorButton, &QColorPushButton::currentColorChanged, [this](const QColor& c) {
    this->OnSetSkyLightBotColor(this->m_lt1gui.m_sbintensitySlider->value(), c);
  });
  QObject::connect(m_lt1gui.m_sbintensitySlider, &QNumericSlider::valueChanged, [this](double v) {
    this->OnSetSkyLightBotColor(v, this->m_lt1gui.m_sbColorButton->GetColor());
  });

  section->setContentLayout(*sectionLayout);
  return section;
}

void
QAppearanceSettingsWidget::OnSetScaleX(double value)
{
  if (!m_scene)
    return;
  m_scene->m_volume->setPhysicalSize(value, m_scene->m_volume->physicalSizeY(), m_scene->m_volume->physicalSizeZ());
  m_scene->initBoundsFromImg(m_scene->m_volume);
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(CameraDirty);
}

void
QAppearanceSettingsWidget::OnSetScaleY(double value)
{
  if (!m_scene)
    return;
  m_scene->m_volume->setPhysicalSize(m_scene->m_volume->physicalSizeX(), value, m_scene->m_volume->physicalSizeZ());
  m_scene->initBoundsFromImg(m_scene->m_volume);
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(CameraDirty);
}

void
QAppearanceSettingsWidget::OnSetScaleZ(double value)
{
  if (!m_scene)
    return;
  m_scene->m_volume->setPhysicalSize(m_scene->m_volume->physicalSizeX(), m_scene->m_volume->physicalSizeY(), value);
  m_scene->initBoundsFromImg(m_scene->m_volume);
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(CameraDirty);
}

void
QAppearanceSettingsWidget::OnSetRoiXMin(int value)
{
  if (!m_scene)
    return;
  glm::vec3 v = m_scene->m_roi.GetMinP();
  v.x = (float)value / 100.0;
  m_scene->m_roi.SetMinP(v);
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(RoiDirty);
}

void
QAppearanceSettingsWidget::OnSetRoiYMin(int value)
{
  if (!m_scene)
    return;
  glm::vec3 v = m_scene->m_roi.GetMinP();
  v.y = (float)value / 100.0;
  m_scene->m_roi.SetMinP(v);
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(RoiDirty);
}

void
QAppearanceSettingsWidget::OnSetRoiZMin(int value)
{
  if (!m_scene)
    return;
  glm::vec3 v = m_scene->m_roi.GetMinP();
  v.z = (float)value / 100.0;
  m_scene->m_roi.SetMinP(v);
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(RoiDirty);
}
void
QAppearanceSettingsWidget::OnSetRoiXMax(int value)
{
  if (!m_scene)
    return;
  glm::vec3 v = m_scene->m_roi.GetMaxP();
  v.x = (float)value / 100.0;
  m_scene->m_roi.SetMaxP(v);
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(RoiDirty);
}
void
QAppearanceSettingsWidget::OnSetRoiYMax(int value)
{
  if (!m_scene)
    return;
  glm::vec3 v = m_scene->m_roi.GetMaxP();
  v.y = (float)value / 100.0;
  m_scene->m_roi.SetMaxP(v);
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(RoiDirty);
}
void
QAppearanceSettingsWidget::OnSetRoiZMax(int value)
{
  if (!m_scene)
    return;
  glm::vec3 v = m_scene->m_roi.GetMaxP();
  v.z = (float)value / 100.0;
  m_scene->m_roi.SetMaxP(v);
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(RoiDirty);
}

void
QAppearanceSettingsWidget::OnSetAreaLightTheta(double value)
{
  if (!m_scene)
    return;
  m_scene->m_lighting.m_Lights[1].m_Theta = value;
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(LightsDirty);
}
void
QAppearanceSettingsWidget::OnSetAreaLightPhi(double value)
{
  if (!m_scene)
    return;
  m_scene->m_lighting.m_Lights[1].m_Phi = value;
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(LightsDirty);
}
void
QAppearanceSettingsWidget::OnSetAreaLightSize(double value)
{
  if (!m_scene)
    return;
  m_scene->m_lighting.m_Lights[1].m_Width = value;
  m_scene->m_lighting.m_Lights[1].m_Height = value;
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(LightsDirty);
}
void
QAppearanceSettingsWidget::OnSetAreaLightDistance(double value)
{
  if (!m_scene)
    return;
  m_scene->m_lighting.m_Lights[1].m_Distance = value;
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(LightsDirty);
}
void
QAppearanceSettingsWidget::OnSetAreaLightColor(double intensity, const QColor& color)
{
  if (!m_scene)
    return;
  qreal rgba[4];
  color.getRgbF(&rgba[0], &rgba[1], &rgba[2], &rgba[3]);

  m_scene->m_lighting.m_Lights[1].m_Color = glm::vec3(rgba[0], rgba[1], rgba[2]);
  m_scene->m_lighting.m_Lights[1].m_ColorIntensity = intensity;
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(LightsDirty);
}

void
QAppearanceSettingsWidget::OnSetSkyLightTopColor(double intensity, const QColor& color)
{
  if (!m_scene)
    return;
  qreal rgba[4];
  color.getRgbF(&rgba[0], &rgba[1], &rgba[2], &rgba[3]);

  m_scene->m_lighting.m_Lights[0].m_ColorTop = glm::vec3(rgba[0], rgba[1], rgba[2]);
  m_scene->m_lighting.m_Lights[0].m_ColorTopIntensity = intensity;
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(LightsDirty);
}
void
QAppearanceSettingsWidget::OnSetSkyLightMidColor(double intensity, const QColor& color)
{
  if (!m_scene)
    return;
  qreal rgba[4];
  color.getRgbF(&rgba[0], &rgba[1], &rgba[2], &rgba[3]);

  m_scene->m_lighting.m_Lights[0].m_ColorMiddle = glm::vec3(rgba[0], rgba[1], rgba[2]);
  m_scene->m_lighting.m_Lights[0].m_ColorMiddleIntensity = intensity;
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(LightsDirty);
}
void
QAppearanceSettingsWidget::OnSetSkyLightBotColor(double intensity, const QColor& color)
{
  if (!m_scene)
    return;
  qreal rgba[4];
  color.getRgbF(&rgba[0], &rgba[1], &rgba[2], &rgba[3]);

  m_scene->m_lighting.m_Lights[0].m_ColorBottom = glm::vec3(rgba[0], rgba[1], rgba[2]);
  m_scene->m_lighting.m_Lights[0].m_ColorBottomIntensity = intensity;
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(LightsDirty);
}

void
QAppearanceSettingsWidget::OnRenderBegin(void)
{
  m_DensityScaleSlider.setValue(m_qrendersettings->GetDensityScale());
  m_ShadingType.setCurrentIndex(m_qrendersettings->GetShadingType());
  m_GradientFactorSlider.setValue(m_qrendersettings->renderSettings()->m_RenderSettings.m_GradientFactor);

  m_StepSizePrimaryRaySlider.setValue(m_qrendersettings->renderSettings()->m_RenderSettings.m_StepSizeFactor, true);
  m_StepSizeSecondaryRaySlider.setValue(m_qrendersettings->renderSettings()->m_RenderSettings.m_StepSizeFactorShadow,
                                        true);
}

void
QAppearanceSettingsWidget::OnSetDensityScale(double DensityScale)
{
  m_qrendersettings->SetDensityScale(DensityScale);
}

void
QAppearanceSettingsWidget::OnSetShadingType(int Index)
{
  m_qrendersettings->SetShadingType(Index);
  m_GradientFactorSlider.setEnabled(Index == 2);
}

void
QAppearanceSettingsWidget::OnSetRendererType(int Index)
{
  m_qrendersettings->SetRendererType(Index);
}

void
QAppearanceSettingsWidget::OnSetGradientFactor(double GradientFactor)
{
  m_qrendersettings->SetGradientFactor(GradientFactor);
}

void
QAppearanceSettingsWidget::OnSetStepSizePrimaryRay(const double& StepSizePrimaryRay)
{
  m_qrendersettings->renderSettings()->m_RenderSettings.m_StepSizeFactor = (float)StepSizePrimaryRay;
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(RenderParamsDirty);
}

void
QAppearanceSettingsWidget::OnSetStepSizeSecondaryRay(const double& StepSizeSecondaryRay)
{
  m_qrendersettings->renderSettings()->m_RenderSettings.m_StepSizeFactorShadow = (float)StepSizeSecondaryRay;
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(RenderParamsDirty);
}

void
QAppearanceSettingsWidget::OnTransferFunctionChanged(void)
{
  m_DensityScaleSlider.setValue(m_qrendersettings->GetDensityScale(), true);
  m_ShadingType.setCurrentIndex(m_qrendersettings->GetShadingType());
  m_GradientFactorSlider.setValue(m_qrendersettings->GetGradientFactor(), true);
}

void
QAppearanceSettingsWidget::OnBackgroundColorChanged(const QColor& color)
{
  if (!m_scene)
    return;
  qreal rgba[4];
  color.getRgbF(&rgba[0], &rgba[1], &rgba[2], &rgba[3]);
  m_scene->m_material.m_backgroundColor[0] = rgba[0];
  m_scene->m_material.m_backgroundColor[1] = rgba[1];
  m_scene->m_material.m_backgroundColor[2] = rgba[2];
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(RenderParamsDirty);
}

void
QAppearanceSettingsWidget::OnDiffuseColorChanged(int i, const QColor& color)
{
  if (!m_scene)
    return;
  qreal rgba[4];
  color.getRgbF(&rgba[0], &rgba[1], &rgba[2], &rgba[3]);
  m_scene->m_material.m_diffuse[i * 3 + 0] = rgba[0];
  m_scene->m_material.m_diffuse[i * 3 + 1] = rgba[1];
  m_scene->m_material.m_diffuse[i * 3 + 2] = rgba[2];
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(RenderParamsDirty);
}

void
QAppearanceSettingsWidget::OnSpecularColorChanged(int i, const QColor& color)
{
  if (!m_scene)
    return;
  qreal rgba[4];
  color.getRgbF(&rgba[0], &rgba[1], &rgba[2], &rgba[3]);
  m_scene->m_material.m_specular[i * 3 + 0] = rgba[0];
  m_scene->m_material.m_specular[i * 3 + 1] = rgba[1];
  m_scene->m_material.m_specular[i * 3 + 2] = rgba[2];
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(RenderParamsDirty);
}

void
QAppearanceSettingsWidget::OnEmissiveColorChanged(int i, const QColor& color)
{
  if (!m_scene)
    return;
  qreal rgba[4];
  color.getRgbF(&rgba[0], &rgba[1], &rgba[2], &rgba[3]);
  m_scene->m_material.m_emissive[i * 3 + 0] = rgba[0];
  m_scene->m_material.m_emissive[i * 3 + 1] = rgba[1];
  m_scene->m_material.m_emissive[i * 3 + 2] = rgba[2];
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(RenderParamsDirty);
}

void
QAppearanceSettingsWidget::OnUpdateLut(int i, const std::vector<LutControlPoint>& stops)
{
  if (!m_scene)
    return;
  m_scene->m_volume->channel((uint32_t)i)->generateFromGradientData(m_scene->m_material.m_gradientData[i]);

  // m_scene->m_volume->channel((uint32_t)i)->generate_controlPoints(stops);
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(TransferFunctionDirty);
}

void
QAppearanceSettingsWidget::OnOpacityChanged(int i, double opacity)
{
  if (!m_scene)
    return;
  // LOG_DEBUG << "window/level: " << window << ", " << level;
  //_scene->_volume->channel((uint32_t)i)->setOpacity(opacity);
  m_scene->m_material.m_opacity[i] = opacity;
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(TransferFunctionDirty);
}

void
QAppearanceSettingsWidget::OnRoughnessChanged(int i, double roughness)
{
  if (!m_scene)
    return;
  m_scene->m_material.m_roughness[i] = roughness;
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(TransferFunctionDirty);
}

void
QAppearanceSettingsWidget::OnChannelChecked(int i, bool is_checked)
{
  if (!m_scene)
    return;
  bool old_value = m_scene->m_material.m_enabled[i];
  if (old_value != is_checked) {
    m_scene->m_material.m_enabled[i] = is_checked;
    m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(VolumeDataDirty);
  }
}

// split color into color and intensity.
inline void
normalizeColorForGui(const glm::vec3& incolor, QColor& outcolor, float& outintensity)
{
  // if any r,g,b is greater than 1, take max value as intensity, else intensity = 1
  float i = std::max(incolor.x, std::max(incolor.y, incolor.z));
  outintensity = (i > 1.0f) ? i : 1.0f;
  glm::vec3 voutcolor = incolor / i;
  outcolor = QColor::fromRgbF(voutcolor.x, voutcolor.y, voutcolor.z);
}

void
QAppearanceSettingsWidget::initLightingControls(Scene* scene)
{
  m_lt0gui.m_thetaSlider->setValue(scene->m_lighting.m_Lights[1].m_Theta);
  m_lt0gui.m_phiSlider->setValue(scene->m_lighting.m_Lights[1].m_Phi);
  m_lt0gui.m_sizeSlider->setValue(scene->m_lighting.m_Lights[1].m_Width);
  m_lt0gui.m_distSlider->setValue(scene->m_lighting.m_Lights[1].m_Distance);
  // split color into color and intensity.
  QColor c;
  float i;
  normalizeColorForGui(scene->m_lighting.m_Lights[1].m_Color, c, i);
  m_lt0gui.m_intensitySlider->setValue(i * scene->m_lighting.m_Lights[1].m_ColorIntensity);
  m_lt0gui.m_areaLightColorButton->SetColor(c);

  normalizeColorForGui(scene->m_lighting.m_Lights[0].m_ColorTop, c, i);
  m_lt1gui.m_stintensitySlider->setValue(i * scene->m_lighting.m_Lights[0].m_ColorTopIntensity);
  m_lt1gui.m_stColorButton->SetColor(c);
  normalizeColorForGui(scene->m_lighting.m_Lights[0].m_ColorMiddle, c, i);
  m_lt1gui.m_smintensitySlider->setValue(i * scene->m_lighting.m_Lights[0].m_ColorMiddleIntensity);
  m_lt1gui.m_smColorButton->SetColor(c);
  normalizeColorForGui(scene->m_lighting.m_Lights[0].m_ColorBottom, c, i);
  m_lt1gui.m_sbintensitySlider->setValue(i * scene->m_lighting.m_Lights[0].m_ColorBottomIntensity);
  m_lt1gui.m_sbColorButton->SetColor(c);
}

void
QAppearanceSettingsWidget::onNewImage(Scene* scene, std::string filepath)
{
  // remove the previous per-channel ui
  for (auto s : m_channelSections) {
    delete s;
  }
  m_channelSections.clear();

  // I don't own this.
  m_scene = scene;
  m_filepath = filepath;

  if (!scene->m_volume) {
    return;
  }

  m_DensityScaleSlider.setValue(m_qrendersettings->renderSettings()->m_RenderSettings.m_DensityScale);
  m_ShadingType.setCurrentIndex(m_qrendersettings->renderSettings()->m_RenderSettings.m_ShadingType);
  m_GradientFactorSlider.setValue(m_qrendersettings->renderSettings()->m_RenderSettings.m_GradientFactor);

  m_StepSizePrimaryRaySlider.setValue(m_qrendersettings->renderSettings()->m_RenderSettings.m_StepSizeFactor);
  m_StepSizeSecondaryRaySlider.setValue(m_qrendersettings->renderSettings()->m_RenderSettings.m_StepSizeFactorShadow);

  QColor cbg = QColor::fromRgbF(m_scene->m_material.m_backgroundColor[0],
                                m_scene->m_material.m_backgroundColor[1],
                                m_scene->m_material.m_backgroundColor[2]);
  m_backgroundColorButton.SetColor(cbg);

  m_roiX->setFirstValue(m_scene->m_roi.GetMinP().x * 100.0);
  m_roiX->setSecondValue(m_scene->m_roi.GetMaxP().x * 100.0);
  m_roiY->setFirstValue(m_scene->m_roi.GetMinP().y * 100.0);
  m_roiY->setSecondValue(m_scene->m_roi.GetMaxP().y * 100.0);
  m_roiZ->setFirstValue(m_scene->m_roi.GetMinP().z * 100.0);
  m_roiZ->setSecondValue(m_scene->m_roi.GetMaxP().z * 100.0);

  m_xscaleSpinner->setValue(m_scene->m_volume->physicalSizeX());
  m_yscaleSpinner->setValue(m_scene->m_volume->physicalSizeY());
  m_zscaleSpinner->setValue(m_scene->m_volume->physicalSizeZ());

  initLightingControls(scene);

  for (uint32_t i = 0; i < scene->m_volume->sizeC(); ++i) {
    bool channelenabled = m_scene->m_material.m_enabled[i];

    Section* section =
      new Section(QString::fromStdString(scene->m_volume->channel(i)->m_name), 0, true, channelenabled);

    auto* fullLayout = new QVBoxLayout();

    auto* sectionLayout = Controls::createFormLayout();

    GradientWidget* editor =
      new GradientWidget(scene->m_volume->channel(i)->m_histogram, &scene->m_material.m_gradientData[i]);
    fullLayout->addWidget(editor);
    // sectionLayout->addRow("Gradient", editor);
    fullLayout->addLayout(sectionLayout);

    QObject::connect(editor, &GradientWidget::gradientStopsChanged, [i, this](const QGradientStops& stops) {
      // convert stops to control points
      std::vector<LutControlPoint> pts;
      for (int i = 0; i < stops.size(); ++i) {
        pts.push_back(LutControlPoint(stops.at(i).first, stops.at(i).second.alphaF()));
      }

      this->OnUpdateLut(i, pts);
    });

    QNumericSlider* opacitySlider = new QNumericSlider();
    opacitySlider->setRange(0.0, 1.0);
    opacitySlider->setSingleStep(0.01);
    opacitySlider->setValue(scene->m_material.m_opacity[i], true);
    sectionLayout->addRow("Opacity", opacitySlider);

    QObject::connect(
      opacitySlider, &QNumericSlider::valueChanged, [i, this](double d) { this->OnOpacityChanged(i, d); });
    // init
    this->OnOpacityChanged(i, scene->m_material.m_opacity[i]);

    QColorPushButton* diffuseColorButton = new QColorPushButton();
    QColor cdiff = QColor::fromRgbF(scene->m_material.m_diffuse[i * 3 + 0],
                                    scene->m_material.m_diffuse[i * 3 + 1],
                                    scene->m_material.m_diffuse[i * 3 + 2]);
    diffuseColorButton->SetColor(cdiff, true);
    sectionLayout->addRow("DiffuseColor", diffuseColorButton);
    QObject::connect(diffuseColorButton, &QColorPushButton::currentColorChanged, [i, this](const QColor& c) {
      this->OnDiffuseColorChanged(i, c);
    });
    // init
    this->OnDiffuseColorChanged(i, cdiff);

    QColorPushButton* specularColorButton = new QColorPushButton();
    QColor cspec = QColor::fromRgbF(scene->m_material.m_specular[i * 3 + 0],
                                    scene->m_material.m_specular[i * 3 + 1],
                                    scene->m_material.m_specular[i * 3 + 2]);
    specularColorButton->SetColor(cspec, true);
    sectionLayout->addRow("SpecularColor", specularColorButton);
    QObject::connect(specularColorButton, &QColorPushButton::currentColorChanged, [i, this](const QColor& c) {
      this->OnSpecularColorChanged(i, c);
    });
    // init
    this->OnSpecularColorChanged(i, cspec);

    QColorPushButton* emissiveColorButton = new QColorPushButton();
    QColor cemis = QColor::fromRgbF(scene->m_material.m_emissive[i * 3 + 0],
                                    scene->m_material.m_emissive[i * 3 + 1],
                                    scene->m_material.m_emissive[i * 3 + 2]);
    emissiveColorButton->SetColor(cemis, true);
    sectionLayout->addRow("EmissiveColor", emissiveColorButton);
    QObject::connect(emissiveColorButton, &QColorPushButton::currentColorChanged, [i, this](const QColor& c) {
      this->OnEmissiveColorChanged(i, c);
    });
    // init
    this->OnEmissiveColorChanged(i, cemis);

    QNumericSlider* roughnessSlider = new QNumericSlider();
    roughnessSlider->setRange(0.0, 100.0);
    roughnessSlider->setSingleStep(0.01);
    roughnessSlider->setValue(scene->m_material.m_roughness[i]);
    sectionLayout->addRow("Glossiness", roughnessSlider);
    QObject::connect(
      roughnessSlider, &QNumericSlider::valueChanged, [i, this](double d) { this->OnRoughnessChanged(i, d); });
    this->OnRoughnessChanged(i, scene->m_material.m_roughness[i]);

    QObject::connect(section, &Section::checked, [i, this](bool is_checked) { this->OnChannelChecked(i, is_checked); });
    this->OnChannelChecked(i, channelenabled);

    section->setContentLayout(*fullLayout);
    // assumes per-channel sections are at the very end of the m_MainLayout
    m_MainLayout.addRow(section);
    m_channelSections.push_back(section);
  }
}
