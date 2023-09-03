#include "AppearanceSettingsWidget.h"
#include "QRenderSettings.h"
#include "RangeWidget.h"
#include "Section.h"

#include "ImageXYZC.h"
#include "renderlib/AppScene.h"
#include "renderlib/Colormap.h"
#include "renderlib/Logging.h"
#include "renderlib/RenderSettings.h"
#include "tfeditor/gradients.h"

#include <QFormLayout>
#include <QLinearGradient>

QGradientStops
colormapToGradient(const std::vector<ColorControlPoint>& v)
{
  QGradientStops stops;
  for (int i = 0; i < v.size(); ++i) {
    stops.push_back(QPair<qreal, QColor>(v[i].first, QColor::fromRgb(v[i].r, v[i].g, v[i].b, v[i].a)));
  }
  return stops;
}

struct GradientSpec
{
  std::vector<ColorControlPoint> stops;
};
static std::map<std::string, GradientSpec> builtInGradients = {
  { "greentored", { { ColorControlPoint(0.0f, 0u, 255u, 0u, 255u), ColorControlPoint(1.0f, 255u, 0u, 0u, 255u) } } },
  { "redtogreen", { { ColorControlPoint(0.0f, 255u, 0u, 0u, 255u), ColorControlPoint(1.0f, 0u, 255u, 0u, 255u) } } }
};

QComboBox*
makeGradientCombo()
{
  QComboBox* cb = new QComboBox();
  const QStringList colorNames = QColor::colorNames();
  int index = 0;
  for (auto& gspec : builtInGradients) {
    QLinearGradient gradient;
    gradient.setStops(colormapToGradient(gspec.second.stops));

    gradient.setStart(0., 0.);     // top left
    gradient.setFinalStop(1., 0.); // bottom right
    gradient.setCoordinateMode(QGradient::ObjectMode);

    QBrush brush(gradient);
    brush.setStyle(Qt::LinearGradientPattern);
    cb->addItem("", QVariant(gspec.first.c_str()));
    const QModelIndex idx = cb->model()->index(index++, 0);
    cb->model()->setData(idx, brush, Qt::BackgroundRole);
  }
  return cb;
}

static const int MAX_CHANNELS_CHECKED = 4;

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

  m_RendererType.setStatusTip(tr("Select volume rendering type"));
  m_RendererType.setToolTip(tr("Select volume rendering type"));

  m_RendererType.addItem("Ray march blending", 0);
  m_RendererType.addItem("Path Traced", 1);

  m_RendererType.setCurrentIndex(1);
  m_MainLayout.addRow("Renderer", &m_RendererType);

  m_DensityScaleSlider.setStatusTip(tr("Set scattering density for volume"));
  m_DensityScaleSlider.setToolTip(tr("Set scattering density for volume"));
  m_DensityScaleSlider.setRange(0.001, 100.0);
  m_DensityScaleSlider.setDecimals(3);
  m_DensityScaleSlider.setValue(rs->m_RenderSettings.m_DensityScale);
  m_MainLayout.addRow("Scattering Density", &m_DensityScaleSlider);

  m_ShadingType.setStatusTip(tr("Select volume shading style"));
  m_ShadingType.setToolTip(tr("Select volume shading style"));
  m_ShadingType.addItem("BRDF Only", 0);
  m_ShadingType.addItem("Phase Function Only", 1);
  m_ShadingType.addItem("Mixed", 2);
  m_ShadingType.setCurrentIndex(rs->m_RenderSettings.m_ShadingType);
  m_MainLayout.addRow("Shading Type", &m_ShadingType);

  m_GradientFactorSlider.setStatusTip(tr("Mix between BRDF and Phase shading"));
  m_GradientFactorSlider.setToolTip(tr("Mix between BRDF and Phase shading"));
  m_GradientFactorSlider.setRange(0.0, 1.0);
  m_GradientFactorSlider.setDecimals(3);
  m_GradientFactorSlider.setValue(rs->m_RenderSettings.m_GradientFactor);
  m_MainLayout.addRow("Shading Type Mixture", &m_GradientFactorSlider);

  QObject::connect(&m_DensityScaleSlider, SIGNAL(valueChanged(double)), this, SLOT(OnSetDensityScale(double)));
  QObject::connect(&m_GradientFactorSlider, SIGNAL(valueChanged(double)), this, SLOT(OnSetGradientFactor(double)));

  m_StepSizePrimaryRaySlider.setStatusTip(tr("Set volume ray march step size for camera rays"));
  m_StepSizePrimaryRaySlider.setToolTip(tr("Set volume ray march step size for camera rays"));
  // step size is in voxels and step sizes of less than 1 voxel are not very useful, while slowing down performance
  m_StepSizePrimaryRaySlider.setRange(1.0, 100.0);
  m_StepSizePrimaryRaySlider.setValue(rs->m_RenderSettings.m_StepSizeFactor);
  m_StepSizePrimaryRaySlider.setDecimals(3);
  m_MainLayout.addRow("Primary Ray Step Size", &m_StepSizePrimaryRaySlider);

  QObject::connect(
    &m_StepSizePrimaryRaySlider, SIGNAL(valueChanged(double)), this, SLOT(OnSetStepSizePrimaryRay(double)));

  m_StepSizeSecondaryRaySlider.setStatusTip(tr("Set volume ray march step size for scattered rays"));
  m_StepSizeSecondaryRaySlider.setToolTip(tr("Set volume ray march step size for scattered rays"));
  m_StepSizeSecondaryRaySlider.setRange(1.0, 100.0);
  m_StepSizeSecondaryRaySlider.setValue(rs->m_RenderSettings.m_StepSizeFactorShadow);
  m_StepSizeSecondaryRaySlider.setDecimals(3);
  m_MainLayout.addRow("Secondary Ray Step Size", &m_StepSizeSecondaryRaySlider);

  QObject::connect(
    &m_StepSizeSecondaryRaySlider, SIGNAL(valueChanged(double)), this, SLOT(OnSetStepSizeSecondaryRay(double)));

  m_backgroundColorButton.setStatusTip(tr("Set background color"));
  m_backgroundColorButton.setToolTip(tr("Set background color"));
  m_backgroundColorButton.SetColor(QColor(0, 0, 0), true);
  m_MainLayout.addRow("Background Color", &m_backgroundColorButton);

  QObject::connect(&m_backgroundColorButton, &QColorPushButton::currentColorChanged, [this](const QColor& c) {
    this->OnBackgroundColorChanged(c);
  });

  auto* bboxLayout = new QHBoxLayout();
  m_showBoundingBoxCheckBox.setChecked(false);
  m_showBoundingBoxCheckBox.setStatusTip(tr("Show/hide bounding box"));
  m_showBoundingBoxCheckBox.setToolTip(tr("Show/hide bounding box"));
  bboxLayout->addWidget(&m_showBoundingBoxCheckBox, 0);

  m_boundingBoxColorButton.setStatusTip(tr("Set bounding box color"));
  m_boundingBoxColorButton.setToolTip(tr("Set bounding box color"));
  m_boundingBoxColorButton.SetColor(QColor(255, 255, 255), true);
  bboxLayout->addWidget(&m_boundingBoxColorButton, 1);

  m_MainLayout.addRow("Bounding Box", bboxLayout);

  QObject::connect(&m_showBoundingBoxCheckBox, &QCheckBox::clicked, [this](const bool is_checked) {
    this->OnShowBoundsChecked(is_checked);
  });
  QObject::connect(&m_boundingBoxColorButton, &QColorPushButton::currentColorChanged, [this](const QColor& c) {
    this->OnBoundingBoxColorChanged(c);
  });

  m_scaleSection = new Section("Volume Scale", 0);
  auto* scaleSectionLayout = new QGridLayout();
  scaleSectionLayout->addWidget(new QLabel("X"), 0, 0);
  m_xscaleSpinner = new QDoubleSpinner();
  m_xscaleSpinner->setStatusTip(tr("Scale volume in X dimension"));
  m_xscaleSpinner->setToolTip(tr("Scale volume in X dimension"));
  m_xscaleSpinner->setDecimals(6);
  m_xscaleSpinner->setValue(1.0);
  scaleSectionLayout->addWidget(m_xscaleSpinner, 0, 1);
  QObject::connect(m_xscaleSpinner,
                   QOverload<double>::of(&QDoubleSpinner::valueChanged),
                   this,
                   &QAppearanceSettingsWidget::OnSetScaleX);
  scaleSectionLayout->addWidget(new QLabel("Y"), 1, 0);
  m_yscaleSpinner = new QDoubleSpinner();
  m_yscaleSpinner->setStatusTip(tr("Scale volume in Y dimension"));
  m_yscaleSpinner->setToolTip(tr("Scale volume in Y dimension"));
  m_yscaleSpinner->setDecimals(6);
  m_yscaleSpinner->setValue(1.0);
  scaleSectionLayout->addWidget(m_yscaleSpinner, 1, 1);
  QObject::connect(m_yscaleSpinner,
                   QOverload<double>::of(&QDoubleSpinner::valueChanged),
                   this,
                   &QAppearanceSettingsWidget::OnSetScaleY);
  scaleSectionLayout->addWidget(new QLabel("Z"), 2, 0);
  m_zscaleSpinner = new QDoubleSpinner();
  m_zscaleSpinner->setStatusTip(tr("Scale volume in Z dimension"));
  m_zscaleSpinner->setToolTip(tr("Scale volume in Z dimension"));
  m_zscaleSpinner->setDecimals(6);
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
  m_roiX->setStatusTip(tr("Set clip planes along X axis"));
  m_roiX->setToolTip(tr("Set clip planes along X axis"));
  m_roiX->setBounds(0, 100);
  m_roiX->setFirstValue(0);
  m_roiX->setSecondValue(100);
  roiSectionLayout->addWidget(m_roiX, 0, 1);
  QObject::connect(m_roiX, &RangeWidget::minValueChanged, this, &QAppearanceSettingsWidget::OnSetRoiXMin);
  QObject::connect(m_roiX, &RangeWidget::maxValueChanged, this, &QAppearanceSettingsWidget::OnSetRoiXMax);
  roiSectionLayout->addWidget(new QLabel("Y"), 1, 0);
  m_roiY = new RangeWidget(Qt::Horizontal);
  m_roiY->setStatusTip(tr("Set clip planes along Y axis"));
  m_roiY->setToolTip(tr("Set clip planes along Y axis"));
  m_roiY->setBounds(0, 100);
  m_roiY->setFirstValue(0);
  m_roiY->setSecondValue(100);
  roiSectionLayout->addWidget(m_roiY, 1, 1);
  QObject::connect(m_roiY, &RangeWidget::minValueChanged, this, &QAppearanceSettingsWidget::OnSetRoiYMin);
  QObject::connect(m_roiY, &RangeWidget::maxValueChanged, this, &QAppearanceSettingsWidget::OnSetRoiYMax);
  roiSectionLayout->addWidget(new QLabel("Z"), 2, 0);
  m_roiZ = new RangeWidget(Qt::Horizontal);
  m_roiZ->setStatusTip(tr("Set clip planes along Z axis"));
  m_roiZ->setToolTip(tr("Set clip planes along Z axis"));
  m_roiZ->setBounds(0, 100);
  m_roiZ->setFirstValue(0);
  m_roiZ->setSecondValue(100);
  roiSectionLayout->addWidget(m_roiZ, 2, 1);
  QObject::connect(m_roiZ, &RangeWidget::minValueChanged, this, &QAppearanceSettingsWidget::OnSetRoiZMin);
  QObject::connect(m_roiZ, &RangeWidget::maxValueChanged, this, &QAppearanceSettingsWidget::OnSetRoiZMax);
  roiSectionLayout->setColumnStretch(0, 1);
  roiSectionLayout->setColumnStretch(1, 3);

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
  m_lt0gui.m_thetaSlider->setStatusTip(tr("Set angle theta for area light"));
  m_lt0gui.m_thetaSlider->setToolTip(tr("Set angle theta for area light"));
  m_lt0gui.m_thetaSlider->setRange(0.0, TWO_PI_F);
  m_lt0gui.m_thetaSlider->setSingleStep(TWO_PI_F / 100.0);
  m_lt0gui.m_thetaSlider->setValue(0.0);
  sectionLayout->addRow("AreaLight Theta", m_lt0gui.m_thetaSlider);
  QObject::connect(
    m_lt0gui.m_thetaSlider, &QNumericSlider::valueChanged, this, &QAppearanceSettingsWidget::OnSetAreaLightTheta);

  m_lt0gui.m_phiSlider = new QNumericSlider();
  m_lt0gui.m_phiSlider->setStatusTip(tr("Set angle phi for area light"));
  m_lt0gui.m_phiSlider->setToolTip(tr("Set angle phi for area light"));
  m_lt0gui.m_phiSlider->setRange(0.0, PI_F);
  m_lt0gui.m_phiSlider->setSingleStep(PI_F / 100.0);
  m_lt0gui.m_phiSlider->setValue(HALF_PI_F);
  sectionLayout->addRow("AreaLight Phi", m_lt0gui.m_phiSlider);
  QObject::connect(
    m_lt0gui.m_phiSlider, &QNumericSlider::valueChanged, this, &QAppearanceSettingsWidget::OnSetAreaLightPhi);

  m_lt0gui.m_sizeSlider = new QNumericSlider();
  m_lt0gui.m_sizeSlider->setStatusTip(tr("Set size for area light"));
  m_lt0gui.m_sizeSlider->setToolTip(tr("Set size for area light"));
  m_lt0gui.m_sizeSlider->setRange(0.1, 5.0);
  m_lt0gui.m_sizeSlider->setSingleStep(5.0 / 100.0);
  m_lt0gui.m_sizeSlider->setValue(1.0);
  sectionLayout->addRow("AreaLight Size", m_lt0gui.m_sizeSlider);
  QObject::connect(
    m_lt0gui.m_sizeSlider, &QNumericSlider::valueChanged, this, &QAppearanceSettingsWidget::OnSetAreaLightSize);

  m_lt0gui.m_distSlider = new QNumericSlider();
  m_lt0gui.m_distSlider->setStatusTip(tr("Set distance for area light"));
  m_lt0gui.m_distSlider->setToolTip(tr("Set distance for area light"));
  m_lt0gui.m_distSlider->setRange(0.1, 100.0);
  m_lt0gui.m_distSlider->setSingleStep(1.0);
  m_lt0gui.m_distSlider->setValue(10.0);
  sectionLayout->addRow("AreaLight Distance", m_lt0gui.m_distSlider);
  QObject::connect(
    m_lt0gui.m_distSlider, &QNumericSlider::valueChanged, this, &QAppearanceSettingsWidget::OnSetAreaLightDistance);

  auto* arealightLayout = new QHBoxLayout();
  m_lt0gui.m_intensitySlider = new QNumericSlider();
  m_lt0gui.m_intensitySlider->setStatusTip(tr("Set intensity for area light"));
  m_lt0gui.m_intensitySlider->setToolTip(tr("Set intensity for area light"));
  m_lt0gui.m_intensitySlider->setRange(0.0, 1000.0);
  m_lt0gui.m_intensitySlider->setSingleStep(1.0);
  m_lt0gui.m_intensitySlider->setValue(100.0);
  arealightLayout->addWidget(m_lt0gui.m_intensitySlider, 1);
  m_lt0gui.m_areaLightColorButton = new QColorPushButton();
  m_lt0gui.m_areaLightColorButton->setStatusTip(tr("Set color for area light"));
  m_lt0gui.m_areaLightColorButton->setToolTip(tr("Set color for area light"));
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
  m_lt1gui.m_stintensitySlider->setStatusTip(tr("Set intensity for top of skylight sphere"));
  m_lt1gui.m_stintensitySlider->setToolTip(tr("Set intensity for top of skylight sphere"));
  m_lt1gui.m_stintensitySlider->setRange(0.0, 10.0);
  m_lt1gui.m_stintensitySlider->setValue(1.0);
  skylightTopLayout->addWidget(m_lt1gui.m_stintensitySlider, 1);
  m_lt1gui.m_stColorButton = new QColorPushButton();
  m_lt1gui.m_stColorButton->setStatusTip(tr("Set color for top of skylight sphere"));
  m_lt1gui.m_stColorButton->setToolTip(tr("Set color for top of skylight sphere"));
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
  m_lt1gui.m_smintensitySlider->setStatusTip(tr("Set intensity for middle of skylight sphere"));
  m_lt1gui.m_smintensitySlider->setToolTip(tr("Set intensity for middle of skylight sphere"));
  m_lt1gui.m_smintensitySlider->setRange(0.0, 10.0);
  m_lt1gui.m_smintensitySlider->setValue(1.0);
  skylightMidLayout->addWidget(m_lt1gui.m_smintensitySlider, 1);
  m_lt1gui.m_smColorButton = new QColorPushButton();
  m_lt1gui.m_smColorButton->setStatusTip(tr("Set color for middle of skylight sphere"));
  m_lt1gui.m_smColorButton->setToolTip(tr("Set color for middle of skylight sphere"));
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
  m_lt1gui.m_sbintensitySlider->setStatusTip(tr("Set intensity for bottom of skylight sphere"));
  m_lt1gui.m_sbintensitySlider->setToolTip(tr("Set intensity for bottom of skylight sphere"));
  m_lt1gui.m_sbintensitySlider->setRange(0.0, 10.0);
  m_lt1gui.m_sbintensitySlider->setValue(1.0);
  skylightBotLayout->addWidget(m_lt1gui.m_sbintensitySlider, 1);
  m_lt1gui.m_sbColorButton = new QColorPushButton();
  m_lt1gui.m_sbColorButton->setStatusTip(tr("Set color for bottom of skylight sphere"));
  m_lt1gui.m_sbColorButton->setToolTip(tr("Set color for bottom of skylight sphere"));
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
  v.x = (float)value / (m_scene->m_volume->sizeX() - 1);
  m_scene->m_roi.SetMinP(v);
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(RoiDirty);
}

void
QAppearanceSettingsWidget::OnSetRoiYMin(int value)
{
  if (!m_scene)
    return;
  glm::vec3 v = m_scene->m_roi.GetMinP();
  v.y = (float)value / (m_scene->m_volume->sizeY() - 1);
  m_scene->m_roi.SetMinP(v);
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(RoiDirty);
}

void
QAppearanceSettingsWidget::OnSetRoiZMin(int value)
{
  if (!m_scene)
    return;
  glm::vec3 v = m_scene->m_roi.GetMinP();
  v.z = (float)value / (m_scene->m_volume->sizeZ() - 1);
  m_scene->m_roi.SetMinP(v);
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(RoiDirty);
}
void
QAppearanceSettingsWidget::OnSetRoiXMax(int value)
{
  if (!m_scene)
    return;
  glm::vec3 v = m_scene->m_roi.GetMaxP();
  v.x = (float)value / (m_scene->m_volume->sizeX() - 1);
  m_scene->m_roi.SetMaxP(v);
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(RoiDirty);
}
void
QAppearanceSettingsWidget::OnSetRoiYMax(int value)
{
  if (!m_scene)
    return;
  glm::vec3 v = m_scene->m_roi.GetMaxP();
  v.y = (float)value / (m_scene->m_volume->sizeY() - 1);
  m_scene->m_roi.SetMaxP(v);
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(RoiDirty);
}
void
QAppearanceSettingsWidget::OnSetRoiZMax(int value)
{
  if (!m_scene)
    return;
  glm::vec3 v = m_scene->m_roi.GetMaxP();
  v.z = (float)value / (m_scene->m_volume->sizeZ() - 1);
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
  float rgba[4];
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
  float rgba[4];
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
  float rgba[4];
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
  float rgba[4];
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
  float rgba[4];
  color.getRgbF(&rgba[0], &rgba[1], &rgba[2], &rgba[3]);
  m_scene->m_material.m_backgroundColor[0] = rgba[0];
  m_scene->m_material.m_backgroundColor[1] = rgba[1];
  m_scene->m_material.m_backgroundColor[2] = rgba[2];
  m_qrendersettings->renderSettings()->m_DirtyFlags.SetFlag(RenderParamsDirty);
}
void
QAppearanceSettingsWidget::OnBoundingBoxColorChanged(const QColor& color)
{
  if (!m_scene)
    return;
  float rgba[4];
  color.getRgbF(&rgba[0], &rgba[1], &rgba[2], &rgba[3]);
  m_scene->m_material.m_boundingBoxColor[0] = rgba[0];
  m_scene->m_material.m_boundingBoxColor[1] = rgba[1];
  m_scene->m_material.m_boundingBoxColor[2] = rgba[2];
}
void
QAppearanceSettingsWidget::OnShowBoundsChecked(bool isChecked)
{
  if (!m_scene)
    return;
  m_scene->m_material.m_showBoundingBox = isChecked;
}

void
QAppearanceSettingsWidget::OnDiffuseColorChanged(int i, const QColor& color)
{
  if (!m_scene)
    return;
  float rgba[4];
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
  float rgba[4];
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
  float rgba[4];
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
QAppearanceSettingsWidget::OnUpdateColormap(int i, const std::vector<ColorControlPoint>& stops)
{
  if (!m_scene)
    return;
  m_scene->m_volume->channel((uint32_t)i)->updateColormap(stops);

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
  // if we are switching one on, count how many sections are checked.
  // if more than 4, then switch this one back off
  if (is_checked) {
    int count = 0;
    for (int j = 0; j < m_channelSections.size(); j++) {
      if (m_channelSections[j]->isChecked())
        count++;
    }
    if (count > MAX_CHANNELS_CHECKED) {
      // uncheck the one that was just checked
      m_channelSections[i]->setChecked(false);
      return;
    }
  }

  // now we can actually update the state
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
QAppearanceSettingsWidget::onNewImage(Scene* scene)
{
  // Don't forget that most ui updating triggered in this function should
  // NOT signal changes to the scene.

  // remove the previous per-channel ui
  for (auto s : m_channelSections) {
    delete s;
  }
  m_channelSections.clear();

  // I don't own this.
  m_scene = scene;

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

  size_t xmax = m_scene->m_volume->sizeX() - 1;
  size_t ymax = m_scene->m_volume->sizeY() - 1;
  size_t zmax = m_scene->m_volume->sizeZ() - 1;

  m_roiX->setBounds(0, xmax, true);
  m_roiY->setBounds(0, ymax, true);
  m_roiZ->setBounds(0, zmax, true);

  m_roiX->setFirstValue(m_scene->m_roi.GetMinP().x * xmax, true);
  m_roiX->setSecondValue(m_scene->m_roi.GetMaxP().x * xmax, true);
  m_roiY->setFirstValue(m_scene->m_roi.GetMinP().y * ymax, true);
  m_roiY->setSecondValue(m_scene->m_roi.GetMaxP().y * ymax, true);
  m_roiZ->setFirstValue(m_scene->m_roi.GetMinP().z * zmax, true);
  m_roiZ->setSecondValue(m_scene->m_roi.GetMaxP().z * zmax, true);

  m_xscaleSpinner->setValue(m_scene->m_volume->physicalSizeX());
  m_yscaleSpinner->setValue(m_scene->m_volume->physicalSizeY());
  m_zscaleSpinner->setValue(m_scene->m_volume->physicalSizeZ());

  QColor cbbox = QColor::fromRgbF(m_scene->m_material.m_boundingBoxColor[0],
                                  m_scene->m_material.m_boundingBoxColor[1],
                                  m_scene->m_material.m_boundingBoxColor[2]);
  m_boundingBoxColorButton.SetColor(cbbox);
  m_showBoundingBoxCheckBox.setChecked(m_scene->m_material.m_showBoundingBox);

  initLightingControls(scene);

  int numEnabled = 0;
  for (uint32_t i = 0; i < scene->m_volume->sizeC(); ++i) {
    bool channelenabled = m_scene->m_material.m_enabled[i];
    // only really allow the first 4 enabled
    if (channelenabled) {
      numEnabled++;
      if (numEnabled > MAX_CHANNELS_CHECKED) {
        channelenabled = false;
        // disable for real!
        m_scene->m_material.m_enabled[i] = false;
      }
    }

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
    this->OnUpdateLut(i, std::vector<LutControlPoint>());

    QNumericSlider* opacitySlider = new QNumericSlider();
    opacitySlider->setStatusTip(tr("Set opacity for channel"));
    opacitySlider->setToolTip(tr("Set opacity for channel"));
    opacitySlider->setRange(0.0, 1.0);
    opacitySlider->setSingleStep(0.01);
    opacitySlider->setValue(scene->m_material.m_opacity[i], true);
    sectionLayout->addRow("Opacity", opacitySlider);

    QObject::connect(
      opacitySlider, &QNumericSlider::valueChanged, [i, this](double d) { this->OnOpacityChanged(i, d); });
    // init
    this->OnOpacityChanged(i, scene->m_material.m_opacity[i]);

    QComboBox* gradients = makeGradientCombo();
    sectionLayout->addRow("ColorMap", gradients);
    QObject::connect(gradients, &QComboBox::currentIndexChanged, [i, gradients, this](int index) {
      LOG_DEBUG << "Selected gradient " << index << " for channel " << i;
      // get string from userdata
      std::string name = gradients->itemData(index).toString().toStdString();
      auto colormap = builtInGradients[name];
      // update channel colormap from stops
      this->OnUpdateColormap(i, colormap.stops);
    });

    QColorPushButton* diffuseColorButton = new QColorPushButton();
    diffuseColorButton->setStatusTip(tr("Set color for channel"));
    diffuseColorButton->setToolTip(tr("Set color for channel"));
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
    specularColorButton->setStatusTip(tr("Set specular color for channel"));
    specularColorButton->setToolTip(tr("Set specular color for channel"));
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
    emissiveColorButton->setStatusTip(tr("Set emissive color for channel"));
    emissiveColorButton->setToolTip(tr("Set emissive color for channel"));
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
    roughnessSlider->setStatusTip(tr("Set specular glossiness for channel"));
    roughnessSlider->setToolTip(tr("Set specular glossiness for channel"));
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
