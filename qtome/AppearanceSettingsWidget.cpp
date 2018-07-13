#include "Stable.h"

#include "AppearanceSettingsWidget.h"
#include "RangeWidget.h"
#include "Section.h"
#include "TransferFunction.h"

#include "ImageXYZC.h"
#include "RenderThread.h"
#include "renderlib/RenderSettings.h"
#include "renderlib/AppScene.h"
#include "renderlib/Logging.h"

QAppearanceSettingsWidget::QAppearanceSettingsWidget(QWidget* pParent, QTransferFunction* tran, RenderSettings* rs) :
	QGroupBox(pParent),
	m_MainLayout(),
	m_DensityScaleSlider(),
	m_RendererType(),
	m_ShadingType(),
	m_GradientFactorLabel(),
	m_GradientFactorSlider(),
	m_StepSizePrimaryRaySlider(),
	m_StepSizeSecondaryRaySlider(),
	_transferFunction(tran),
	_scene(nullptr)
{
	setLayout(&m_MainLayout);

	m_MainLayout.addWidget(new QLabel("Renderer"), 1, 0);
	m_RendererType.addItem("OpenGL simple", 0);
	m_RendererType.addItem("CUDA full", 1);
	m_RendererType.setCurrentIndex(1);
	m_MainLayout.addWidget(&m_RendererType, 1, 1, 1, 2);

	m_MainLayout.addWidget(new QLabel("Density Scale"), 2, 0);
	m_DensityScaleSlider.setRange(0.001, 100.0);
	m_DensityScaleSlider.setDecimals(3);
	m_DensityScaleSlider.setValue(rs->m_RenderSettings.m_DensityScale);
	m_MainLayout.addWidget(&m_DensityScaleSlider, 2, 1, 1, 2);

	m_MainLayout.addWidget(new QLabel("Shading Type"), 3, 0);

	m_ShadingType.addItem("BRDF Only", 0);
	m_ShadingType.addItem("Phase Function Only", 1);
	m_ShadingType.addItem("Hybrid", 2);
	m_MainLayout.addWidget(&m_ShadingType, 3, 1, 1, 2);

	m_GradientFactorLabel.setText("Gradient Factor");
	m_MainLayout.addWidget(&m_GradientFactorLabel, 4, 0);
	m_GradientFactorSlider.setRange(0.001, 100.0);
	m_GradientFactorSlider.setDecimals(3);
	m_GradientFactorSlider.setValue(rs->m_RenderSettings.m_GradientFactor);
	m_MainLayout.addWidget(&m_GradientFactorSlider, 4, 1, 1, 2);

	QObject::connect(&m_DensityScaleSlider, SIGNAL(valueChanged(double)), this, SLOT(OnSetDensityScale(double)));
	QObject::connect(&m_GradientFactorSlider, SIGNAL(valueChanged(double)), this, SLOT(OnSetGradientFactor(double)));

	m_MainLayout.addWidget(new QLabel("Primary Step Size"), 5, 0);
	m_StepSizePrimaryRaySlider.setRange(1.0, 10.0);
	m_StepSizePrimaryRaySlider.setDecimals(2);
	m_MainLayout.addWidget(&m_StepSizePrimaryRaySlider, 5, 1, 1, 2);

	QObject::connect(&m_StepSizePrimaryRaySlider, SIGNAL(valueChanged(double)), this, SLOT(OnSetStepSizePrimaryRay(double)));

	m_MainLayout.addWidget(new QLabel("Secondary Step Size"), 6, 0);

	m_StepSizeSecondaryRaySlider.setRange(1.0, 10.0);
	m_StepSizeSecondaryRaySlider.setDecimals(2);

	m_MainLayout.addWidget(&m_StepSizeSecondaryRaySlider, 6, 1, 1, 2);

	QObject::connect(&m_StepSizeSecondaryRaySlider, SIGNAL(valueChanged(double)), this, SLOT(OnSetStepSizeSecondaryRay(double)));

	_scaleSection = new Section("Scale", 0);
	auto* scaleSectionLayout = new QGridLayout();
	scaleSectionLayout->addWidget(new QLabel("X"), 0, 0);
	QDoubleSpinner* xscaleSpinner = new QDoubleSpinner();
	scaleSectionLayout->addWidget(xscaleSpinner, 0, 1);
	QObject::connect(xscaleSpinner, QOverload<double>::of(&QDoubleSpinner::valueChanged), this, &QAppearanceSettingsWidget::OnSetScaleX);
	scaleSectionLayout->addWidget(new QLabel("Y"), 1, 0);
	QDoubleSpinner* yscaleSpinner = new QDoubleSpinner();
	scaleSectionLayout->addWidget(yscaleSpinner, 1, 1);
	QObject::connect(yscaleSpinner, QOverload<double>::of(&QDoubleSpinner::valueChanged), this, &QAppearanceSettingsWidget::OnSetScaleY);
	scaleSectionLayout->addWidget(new QLabel("Z"), 2, 0);
	QDoubleSpinner* zscaleSpinner = new QDoubleSpinner();
	scaleSectionLayout->addWidget(zscaleSpinner, 2, 1);
	QObject::connect(zscaleSpinner, QOverload<double>::of(&QDoubleSpinner::valueChanged), this, &QAppearanceSettingsWidget::OnSetScaleZ);

	_scaleSection->setContentLayout(*scaleSectionLayout);
	m_MainLayout.addWidget(_scaleSection, 12, 0, 1, -1);



	_clipRoiSection = new Section("ROI", 0);
	auto* roiSectionLayout = new QGridLayout();
	roiSectionLayout->addWidget(new QLabel("X"), 0, 0);
	RangeWidget* xSlider = new RangeWidget(Qt::Horizontal);
	xSlider->setRange(0, 100);
	xSlider->setFirstValue(0);
	xSlider->setSecondValue(100);
	roiSectionLayout->addWidget(xSlider, 0, 1);
	QObject::connect(xSlider, &RangeWidget::firstValueChanged, this, &QAppearanceSettingsWidget::OnSetRoiXMin);
	QObject::connect(xSlider, &RangeWidget::secondValueChanged, this, &QAppearanceSettingsWidget::OnSetRoiXMax);
	roiSectionLayout->addWidget(new QLabel("Y"), 1, 0);
	RangeWidget* ySlider = new RangeWidget(Qt::Horizontal);
	ySlider->setRange(0, 100);
	ySlider->setFirstValue(0);
	ySlider->setSecondValue(100);
	roiSectionLayout->addWidget(ySlider, 1, 1);
	QObject::connect(ySlider, &RangeWidget::firstValueChanged, this, &QAppearanceSettingsWidget::OnSetRoiYMin);
	QObject::connect(ySlider, &RangeWidget::secondValueChanged, this, &QAppearanceSettingsWidget::OnSetRoiYMax);
	roiSectionLayout->addWidget(new QLabel("Z"), 2, 0);
	RangeWidget* zSlider = new RangeWidget(Qt::Horizontal);
	zSlider->setRange(0, 100);
	zSlider->setFirstValue(0);
	zSlider->setSecondValue(100);
	roiSectionLayout->addWidget(zSlider, 2, 1);
	QObject::connect(zSlider, &RangeWidget::firstValueChanged, this, &QAppearanceSettingsWidget::OnSetRoiZMin);
	QObject::connect(zSlider, &RangeWidget::secondValueChanged, this, &QAppearanceSettingsWidget::OnSetRoiZMax);

	_clipRoiSection->setContentLayout(*roiSectionLayout);
	m_MainLayout.addWidget(_clipRoiSection, 13, 0, 1, -1);


	Section* section = new Section("Lighting", 0);
	auto* sectionLayout = new QGridLayout();

	int row = 0;
	sectionLayout->addWidget(new QLabel("AreaLight Theta"), row, 0);
	QNumericSlider* thetaSlider = new QNumericSlider();
	thetaSlider->setRange(0.0, 3.14159265*2.0);
	thetaSlider->setValue(0.0);
	sectionLayout->addWidget(thetaSlider, row, 1, 1, 4);
	QObject::connect(thetaSlider, &QNumericSlider::valueChanged, this, &QAppearanceSettingsWidget::OnSetAreaLightTheta);

	row++;
	sectionLayout->addWidget(new QLabel("AreaLight Phi"), row, 0);
	QNumericSlider* phiSlider = new QNumericSlider();
	phiSlider->setRange(0.0, 3.14159265);
	phiSlider->setValue(0.0);
	sectionLayout->addWidget(phiSlider, row, 1, 1, 4);
	QObject::connect(phiSlider, &QNumericSlider::valueChanged, this, &QAppearanceSettingsWidget::OnSetAreaLightPhi);

	row++;
	sectionLayout->addWidget(new QLabel("AreaLight Size"), row, 0);
	QNumericSlider* sizeSlider = new QNumericSlider();
	sizeSlider->setRange(0.1, 5.0);
	sizeSlider->setValue(1.0);
	sectionLayout->addWidget(sizeSlider, row, 1, 1, 4);
	QObject::connect(sizeSlider, &QNumericSlider::valueChanged, this, &QAppearanceSettingsWidget::OnSetAreaLightSize);

	row++;
	sectionLayout->addWidget(new QLabel("AreaLight Distance"), row, 0);
	QNumericSlider* distSlider = new QNumericSlider();
	distSlider->setRange(0.1, 100.0);
	distSlider->setValue(10.0);
	sectionLayout->addWidget(distSlider, row, 1, 1, 4);
	QObject::connect(distSlider, &QNumericSlider::valueChanged, this, &QAppearanceSettingsWidget::OnSetAreaLightDistance);

	row++;
	sectionLayout->addWidget(new QLabel("AreaLight Intensity"), row, 0);
	QNumericSlider* intensitySlider = new QNumericSlider();
	intensitySlider->setRange(0.1, 1000.0);
	intensitySlider->setValue(100.0);
	sectionLayout->addWidget(intensitySlider, row, 1, 1, 3);
	QColorPushButton* areaLightColorButton = new QColorPushButton();
	sectionLayout->addWidget(areaLightColorButton, row, 4);
	QObject::connect(areaLightColorButton, &QColorPushButton::currentColorChanged,
		[this, intensitySlider](const QColor& c) { this->OnSetAreaLightColor(intensitySlider->value(), c); });
	QObject::connect(intensitySlider, &QNumericSlider::valueChanged,
		[this, areaLightColorButton](double v) { this->OnSetAreaLightColor(v, areaLightColorButton->GetColor()); });

	row++;
	sectionLayout->addWidget(new QLabel("SkyLight Top"), row, 0);
	QNumericSlider* stintensitySlider = new QNumericSlider();
	stintensitySlider->setRange(0.1, 10.0);
	stintensitySlider->setValue(1.0);
	sectionLayout->addWidget(stintensitySlider, row, 1, 1, 3);
	QColorPushButton* stColorButton = new QColorPushButton();
	sectionLayout->addWidget(stColorButton, row, 4);
	QObject::connect(stColorButton, &QColorPushButton::currentColorChanged,
		[this, stintensitySlider](const QColor& c) { this->OnSetSkyLightTopColor(stintensitySlider->value(), c); });
	QObject::connect(stintensitySlider, &QNumericSlider::valueChanged,
		[this, stColorButton](double v) { this->OnSetSkyLightTopColor(v, stColorButton->GetColor()); });

	row++;
	sectionLayout->addWidget(new QLabel("SkyLight Mid"), row, 0);
	QNumericSlider* smintensitySlider = new QNumericSlider();
	smintensitySlider->setRange(0.1, 10.0);
	smintensitySlider->setValue(1.0);
	sectionLayout->addWidget(smintensitySlider, row, 1, 1, 3);
	QColorPushButton* smColorButton = new QColorPushButton();
	sectionLayout->addWidget(smColorButton, row, 4);
	QObject::connect(smColorButton, &QColorPushButton::currentColorChanged,
		[this, smintensitySlider](const QColor& c) { this->OnSetSkyLightMidColor(smintensitySlider->value(), c); });
	QObject::connect(smintensitySlider, &QNumericSlider::valueChanged,
		[this, smColorButton](double v) { this->OnSetSkyLightMidColor(v, smColorButton->GetColor()); });

	row++;
	sectionLayout->addWidget(new QLabel("SkyLight Bot"), row, 0);
	QNumericSlider* sbintensitySlider = new QNumericSlider();
	sbintensitySlider->setRange(0.1, 10.0);
	sbintensitySlider->setValue(1.0);
	sectionLayout->addWidget(sbintensitySlider, row, 1, 1, 3);
	QColorPushButton* sbColorButton = new QColorPushButton();
	sectionLayout->addWidget(sbColorButton, row, 4);
	QObject::connect(sbColorButton, &QColorPushButton::currentColorChanged,
		[this, sbintensitySlider](const QColor& c) { this->OnSetSkyLightBotColor(sbintensitySlider->value(), c); });
	QObject::connect(sbintensitySlider, &QNumericSlider::valueChanged,
		[this, sbColorButton](double v) { this->OnSetSkyLightBotColor(v, sbColorButton->GetColor()); });


	section->setContentLayout(*sectionLayout);
	m_MainLayout.addWidget(section, 14, 0, 1, -1);

	QObject::connect(&m_RendererType, SIGNAL(currentIndexChanged(int)), this, SLOT(OnSetRendererType(int)));
	QObject::connect(&m_ShadingType, SIGNAL(currentIndexChanged(int)), this, SLOT(OnSetShadingType(int)));
	//QObject::connect(&gStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	
	QObject::connect(_transferFunction, SIGNAL(Changed()), this, SLOT(OnTransferFunctionChanged()));

}

void QAppearanceSettingsWidget::OnSetScaleX(double value)
{
	if (!_scene) return;
	_scene->_volume->setPhysicalSize(value, _scene->_volume->physicalSizeY(), _scene->_volume->physicalSizeZ());
	_scene->initSceneFromImg(_scene->_volume);
	_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(CameraDirty);
}
void QAppearanceSettingsWidget::OnSetScaleY(double value)
{
	if (!_scene) return;
	_scene->_volume->setPhysicalSize(_scene->_volume->physicalSizeX(), value, _scene->_volume->physicalSizeZ());
	_scene->initSceneFromImg(_scene->_volume);
	_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(CameraDirty);
}
void QAppearanceSettingsWidget::OnSetScaleZ(double value)
{
	if (!_scene) return;
	_scene->_volume->setPhysicalSize(_scene->_volume->physicalSizeX(), _scene->_volume->physicalSizeY(), value);
	_scene->initSceneFromImg(_scene->_volume);
	_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(CameraDirty);
}
void QAppearanceSettingsWidget::OnSetRoiXMin(int value)
{
	if (!_scene) return;
	glm::vec3 v = _scene->_roi.GetMinP();
	v.x = (float)value / 100.0;
	_scene->_roi.SetMinP(v);
	_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(RoiDirty);
}
void QAppearanceSettingsWidget::OnSetRoiYMin(int value)
{
	if (!_scene) return;
	glm::vec3 v = _scene->_roi.GetMinP();
	v.y = (float)value / 100.0;
	_scene->_roi.SetMinP(v);
	_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(RoiDirty);
}
void QAppearanceSettingsWidget::OnSetRoiZMin(int value)
{
	if (!_scene) return;
	glm::vec3 v = _scene->_roi.GetMinP();
	v.z = (float)value / 100.0;
	_scene->_roi.SetMinP(v);
	_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(RoiDirty);
}
void QAppearanceSettingsWidget::OnSetRoiXMax(int value)
{
	if (!_scene) return;
	glm::vec3 v = _scene->_roi.GetMaxP();
	v.x = (float)value / 100.0;
	_scene->_roi.SetMaxP(v);
	_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(RoiDirty);
}
void QAppearanceSettingsWidget::OnSetRoiYMax(int value)
{
	if (!_scene) return;
	glm::vec3 v = _scene->_roi.GetMaxP();
	v.y = (float)value / 100.0;
	_scene->_roi.SetMaxP(v);
	_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(RoiDirty);
}
void QAppearanceSettingsWidget::OnSetRoiZMax(int value)
{
	if (!_scene) return;
	glm::vec3 v = _scene->_roi.GetMaxP();
	v.z = (float)value / 100.0;
	_scene->_roi.SetMaxP(v);
	_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(RoiDirty);
}

void QAppearanceSettingsWidget::OnSetAreaLightTheta(double value)
{
	if (!_scene) return;
	_scene->_lighting.m_Lights[1].m_Theta = value;
	_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(LightsDirty);
}
void QAppearanceSettingsWidget::OnSetAreaLightPhi(double value)
{
	if (!_scene) return;
	_scene->_lighting.m_Lights[1].m_Phi = value;
	_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(LightsDirty);
}
void QAppearanceSettingsWidget::OnSetAreaLightSize(double value)
{
	if (!_scene) return;
	_scene->_lighting.m_Lights[1].m_Width = value;
	_scene->_lighting.m_Lights[1].m_Height = value;
	_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(LightsDirty);
}
void QAppearanceSettingsWidget::OnSetAreaLightDistance(double value)
{
	if (!_scene) return;
	_scene->_lighting.m_Lights[1].m_Distance = value;
	_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(LightsDirty);
}
void QAppearanceSettingsWidget::OnSetAreaLightColor(double intensity, const QColor& color)
{
	if (!_scene) return;
	qreal rgba[4];
	color.getRgbF(&rgba[0], &rgba[1], &rgba[2], &rgba[3]);

	_scene->_lighting.m_Lights[1].m_Color = glm::vec3(intensity * rgba[0], intensity*rgba[1], intensity*rgba[2]);
	_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(LightsDirty);
}

void QAppearanceSettingsWidget::OnSetSkyLightTopColor(double intensity, const QColor& color)
{
	if (!_scene) return;
	qreal rgba[4];
	color.getRgbF(&rgba[0], &rgba[1], &rgba[2], &rgba[3]);

	_scene->_lighting.m_Lights[0].m_ColorTop = glm::vec3(intensity * rgba[0], intensity*rgba[1], intensity*rgba[2]);
	_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(LightsDirty);
}
void QAppearanceSettingsWidget::OnSetSkyLightMidColor(double intensity, const QColor& color)
{
	if (!_scene) return;
	qreal rgba[4];
	color.getRgbF(&rgba[0], &rgba[1], &rgba[2], &rgba[3]);

	_scene->_lighting.m_Lights[0].m_ColorMiddle = glm::vec3(intensity * rgba[0], intensity*rgba[1], intensity*rgba[2]);
	_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(LightsDirty);
}
void QAppearanceSettingsWidget::OnSetSkyLightBotColor(double intensity, const QColor& color)
{
	if (!_scene) return;
	qreal rgba[4];
	color.getRgbF(&rgba[0], &rgba[1], &rgba[2], &rgba[3]);

	_scene->_lighting.m_Lights[0].m_ColorBottom = glm::vec3(intensity * rgba[0], intensity*rgba[1], intensity*rgba[2]);
	_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(LightsDirty);
}

void QAppearanceSettingsWidget::OnRenderBegin(void)
{
	m_DensityScaleSlider.setValue(_transferFunction->GetDensityScale());
	m_ShadingType.setCurrentIndex(_transferFunction->GetShadingType());
	m_GradientFactorSlider.setValue(_transferFunction->renderSettings()->m_RenderSettings.m_GradientFactor);

	m_StepSizePrimaryRaySlider.setValue(_transferFunction->renderSettings()->m_RenderSettings.m_StepSizeFactor, true);
	m_StepSizeSecondaryRaySlider.setValue(_transferFunction->renderSettings()->m_RenderSettings.m_StepSizeFactorShadow, true);
}

void QAppearanceSettingsWidget::OnSetDensityScale(double DensityScale)
{
	_transferFunction->SetDensityScale(DensityScale);
}

void QAppearanceSettingsWidget::OnSetShadingType(int Index)
{
	_transferFunction->SetShadingType(Index);
	m_GradientFactorLabel.setEnabled(Index == 2);
	m_GradientFactorSlider.setEnabled(Index == 2);
}

void QAppearanceSettingsWidget::OnSetRendererType(int Index)
{
	_transferFunction->SetRendererType(Index);
}

void QAppearanceSettingsWidget::OnSetGradientFactor(double GradientFactor)
{
	_transferFunction->SetGradientFactor(GradientFactor);
}

void QAppearanceSettingsWidget::OnSetStepSizePrimaryRay(const double& StepSizePrimaryRay)
{
	_transferFunction->renderSettings()->m_RenderSettings.m_StepSizeFactor = (float)StepSizePrimaryRay;
	_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(RenderParamsDirty);
}

void QAppearanceSettingsWidget::OnSetStepSizeSecondaryRay(const double& StepSizeSecondaryRay)
{
	_transferFunction->renderSettings()->m_RenderSettings.m_StepSizeFactorShadow = (float)StepSizeSecondaryRay;
	_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(RenderParamsDirty);
}

void QAppearanceSettingsWidget::OnTransferFunctionChanged(void)
{
	m_DensityScaleSlider.setValue(_transferFunction->GetDensityScale(), true);
	m_ShadingType.setCurrentIndex(_transferFunction->GetShadingType());
	m_GradientFactorSlider.setValue(_transferFunction->GetGradientFactor(), true);
}

void QAppearanceSettingsWidget::OnDiffuseColorChanged(int i, const QColor& color)
{
	if (!_scene) return;
	qreal rgba[4];
	color.getRgbF(&rgba[0], &rgba[1], &rgba[2], &rgba[3]);
	_scene->_material.diffuse[i * 3 + 0] = rgba[0];
	_scene->_material.diffuse[i * 3 + 1] = rgba[1];
	_scene->_material.diffuse[i * 3 + 2] = rgba[2];
	_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(RenderParamsDirty);
}

void QAppearanceSettingsWidget::OnSpecularColorChanged(int i, const QColor& color)
{
	if (!_scene) return;
	qreal rgba[4];
	color.getRgbF(&rgba[0], &rgba[1], &rgba[2], &rgba[3]);
	_scene->_material.specular[i * 3 + 0] = rgba[0];
	_scene->_material.specular[i * 3 + 1] = rgba[1];
	_scene->_material.specular[i * 3 + 2] = rgba[2];
	_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(RenderParamsDirty);
}

void QAppearanceSettingsWidget::OnEmissiveColorChanged(int i, const QColor& color)
{
	if (!_scene) return;
	qreal rgba[4];
	color.getRgbF(&rgba[0], &rgba[1], &rgba[2], &rgba[3]);
	_scene->_material.emissive[i * 3 + 0] = rgba[0];
	_scene->_material.emissive[i * 3 + 1] = rgba[1];
	_scene->_material.emissive[i * 3 + 2] = rgba[2];
	_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(RenderParamsDirty);
}
void QAppearanceSettingsWidget::OnSetWindowLevel(int i, double window, double level)
{
	if (!_scene) return;
	//LOG_DEBUG << "window/level: " << window << ", " << level;
	_scene->_volume->channel((uint32_t)i)->generate_windowLevel(window, level);

	_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(TransferFunctionDirty);
}

void QAppearanceSettingsWidget::OnOpacityChanged(int i, double opacity)
{
	if (!_scene) return;
	//LOG_DEBUG << "window/level: " << window << ", " << level;
	//_scene->_volume->channel((uint32_t)i)->setOpacity(opacity);
	_scene->_material.opacity[i] = opacity;
	_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(TransferFunctionDirty);
}

void QAppearanceSettingsWidget::OnRoughnessChanged(int i, double roughness)
{
	if (!_scene) return;
	_scene->_material.roughness[i] = roughness;
	_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(TransferFunctionDirty);
}

void QAppearanceSettingsWidget::OnChannelChecked(int i, bool is_checked) {
	if (!_scene) return;
	_scene->_material.enabled[i] = is_checked;
	_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(VolumeDataDirty);
}

inline QVector<QColor> rndColors(int count) {
	QVector<QColor> colors;
	colors.push_back(QColor(255, 0, 255));
	colors.push_back(QColor(255, 255, 255));
	colors.push_back(QColor(0, 255, 255));
	float currentHue = 0.0;
	for (int i = 0; i < count; i++) {
		colors.push_back(QColor::fromHslF(currentHue, 1.0, 0.5));
		currentHue += 0.618033988749895f;
		currentHue = std::fmod(currentHue, 1.0f);
	}
	return colors;
}

void QAppearanceSettingsWidget::onNewImage(Scene* scene)
{
	// remove the previous per-channel ui
	for (auto s: _channelSections) {
		delete s;
	}
	_channelSections.clear();

	// I don't own this.
	_scene = scene;

	QVector<QColor> colors = rndColors(scene->_volume->sizeC());

	for (uint32_t i = 0; i < scene->_volume->sizeC(); ++i) {
		// first 3 channels will be chekced
		bool channelenabled = (i < 3);
		Section* section = new Section(scene->_volume->channel(i)->_name, 0, channelenabled);

		auto* sectionLayout = new QGridLayout();

		float init_window, init_level;
		scene->_volume->channel(i)->generate_auto2(init_window, init_level);

		int row = 0;
		sectionLayout->addWidget(new QLabel("Window"), row, 0);
		QNumericSlider* windowSlider = new QNumericSlider();
		windowSlider->setRange(0.001, 1.0);
		windowSlider->setValue(init_window, true);
		sectionLayout->addWidget(windowSlider, row, 1, 1, 2);

		row++;
		sectionLayout->addWidget(new QLabel("Level"), row, 0);
		QNumericSlider* levelSlider = new QNumericSlider();
		levelSlider->setRange(0.001, 1.0);
		levelSlider->setValue(init_level, true);
		sectionLayout->addWidget(levelSlider, row, 1, 1, 2);

		QObject::connect(windowSlider, &QNumericSlider::valueChanged, [i, this, levelSlider](double d) {
			this->OnSetWindowLevel(i, d, levelSlider->value());
		});
		QObject::connect(levelSlider, &QNumericSlider::valueChanged, [i, this, windowSlider](double d) {
			this->OnSetWindowLevel(i, windowSlider->value(), d);
		});
		// init
		//this->OnSetWindowLevel(i, init_window, init_level);
		row++;
		QPushButton* autoButton = new QPushButton("Auto");
		sectionLayout->addWidget(autoButton, row, 0);
		QObject::connect(autoButton, &QPushButton::clicked, [this, i, windowSlider, levelSlider]() {
			float w, l;
			this->_scene->_volume->channel((uint32_t)i)->generate_auto(w,l);
			//LOG_DEBUG << "Window/level: " << w << " , " << l;
			windowSlider->setValue(w, true);
			levelSlider->setValue(l, true);
			this->_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(TransferFunctionDirty);
		});
		QPushButton* auto2Button = new QPushButton("Auto2");
		sectionLayout->addWidget(auto2Button, row, 1);
		QObject::connect(auto2Button, &QPushButton::clicked, [this, i, windowSlider, levelSlider]() {
			float w, l;
			this->_scene->_volume->channel((uint32_t)i)->generate_auto2(w, l);
			//LOG_DEBUG << "Window/level: " << w << " , " << l;
			windowSlider->setValue(w, true);
			levelSlider->setValue(l, true);
			this->_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(TransferFunctionDirty);
		});
		QPushButton* bestfitButton = new QPushButton("BestFit");
		sectionLayout->addWidget(bestfitButton, row, 2);
		QObject::connect(bestfitButton, &QPushButton::clicked, [this, i, windowSlider, levelSlider]() {
			float w, l;
			this->_scene->_volume->channel((uint32_t)i)->generate_bestFit(w, l);
			windowSlider->setValue(w, true);
			levelSlider->setValue(l, true);
			//LOG_DEBUG << "Window/level: " << w << " , " << l;
			this->_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(TransferFunctionDirty);
		});
		QPushButton* chimeraxButton = new QPushButton("ChimX");
		sectionLayout->addWidget(chimeraxButton, row, 3);
		QObject::connect(chimeraxButton, &QPushButton::clicked, [this, i]() {
			this->_scene->_volume->channel((uint32_t)i)->generate_chimerax();
			this->_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(TransferFunctionDirty);
		});
		QPushButton* eqButton = new QPushButton("Eq");
		sectionLayout->addWidget(eqButton, row, 4);
		QObject::connect(eqButton, &QPushButton::clicked, [this, i]() {
			this->_scene->_volume->channel((uint32_t)i)->generate_equalized();
			this->_transferFunction->renderSettings()->m_DirtyFlags.SetFlag(TransferFunctionDirty);
		});

		row++;
		sectionLayout->addWidget(new QLabel("Opacity"), row, 0);
		QNumericSlider* opacitySlider = new QNumericSlider();
		opacitySlider->setRange(0.0, 1.0);
		opacitySlider->setValue(1.0, true);
		sectionLayout->addWidget(opacitySlider, row, 1, 1, 2);

		QObject::connect(opacitySlider, &QNumericSlider::valueChanged, [i, this](double d) {
			this->OnOpacityChanged(i, d);
		});
		// init
		this->OnOpacityChanged(i, 1.0);

		row++;
		QColorPushButton* diffuseColorButton = new QColorPushButton();
		diffuseColorButton->SetColor(colors[i], true);
		sectionLayout->addWidget(new QLabel("DiffuseColor"), row, 0);
		sectionLayout->addWidget(diffuseColorButton, row, 2);
		QObject::connect(diffuseColorButton, &QColorPushButton::currentColorChanged, [i, this](const QColor& c) {
			this->OnDiffuseColorChanged(i, c);
		});
		// init
		this->OnDiffuseColorChanged(i, colors[i]);

		row++;
		QColorPushButton* specularColorButton = new QColorPushButton();
		specularColorButton->SetColor(QColor::fromRgbF(0.0f, 0.0f, 0.0f), true);
		sectionLayout->addWidget(new QLabel("SpecularColor"), row, 0);
		sectionLayout->addWidget(specularColorButton, row, 2);
		QObject::connect(specularColorButton, &QColorPushButton::currentColorChanged, [i, this](const QColor& c) {
			this->OnSpecularColorChanged(i, c);
		});
		// init
		this->OnSpecularColorChanged(i, QColor::fromRgbF(0.0f, 0.0f, 0.0f));

		row++;
		QColorPushButton* emissiveColorButton = new QColorPushButton();
		emissiveColorButton->SetColor(QColor::fromRgbF(0.0f, 0.0f, 0.0f), true);
		sectionLayout->addWidget(new QLabel("EmissiveColor"), row, 0);
		sectionLayout->addWidget(emissiveColorButton, row, 2);
		QObject::connect(emissiveColorButton, &QColorPushButton::currentColorChanged, [i, this](const QColor& c) {
			this->OnEmissiveColorChanged(i, c);
		});
		// init
		this->OnEmissiveColorChanged(i, QColor::fromRgbF(0.0f, 0.0f, 0.0f));

		row++;
		sectionLayout->addWidget(new QLabel("Glossiness"), row, 0);
		QNumericSlider* roughnessSlider = new QNumericSlider();
		roughnessSlider->setRange(0.0, 100.0);
		roughnessSlider->setValue(0.0);
		sectionLayout->addWidget(roughnessSlider, row, 1, 1, 2);
		QObject::connect(roughnessSlider, &QNumericSlider::valueChanged, [i, this](double d) {
			this->OnRoughnessChanged(i, d);
		});
		this->OnRoughnessChanged(i, 0.0);

		QObject::connect(section, &Section::checked, [i, this](bool is_checked) {
			this->OnChannelChecked(i, is_checked);
		});
		this->OnChannelChecked(i, channelenabled);

		section->setContentLayout(*sectionLayout);
		m_MainLayout.addWidget(section, 15+i, 0, 1, -1);
		_channelSections.push_back(section);
	}
}
