#include "Stable.h"

#include "AppearanceSettingsWidget.h"
#include "Section.h"
#include "TransferFunction.h"

#include "ImageXYZC.h"
#include "RenderThread.h"
#include "Scene.h"
#include "AppScene.h"

QAppearanceSettingsWidget::QAppearanceSettingsWidget(QWidget* pParent, QTransferFunction* tran, CScene* scene) :
	QGroupBox(pParent),
	m_MainLayout(),
	m_DensityScaleSlider(),
	m_DensityScaleSpinner(),
	m_RendererType(),
	m_ShadingType(),
	m_GradientFactorLabel(),
	m_GradientFactorSlider(),
	m_GradientFactorSpinner(),
	m_StepSizePrimaryRaySlider(),
	m_StepSizePrimaryRaySpinner(),
	m_StepSizeSecondaryRaySlider(),
	m_StepSizeSecondaryRaySpinner(),
	_transferFunction(tran)
{
	setLayout(&m_MainLayout);

	m_MainLayout.addWidget(new QLabel("Renderer"), 1, 0);
	m_RendererType.addItem("OpenGL simple", 0);
	m_RendererType.addItem("CUDA full", 1);
	m_RendererType.setCurrentIndex(1);
	m_MainLayout.addWidget(&m_RendererType, 1, 1, 1, 2);

	m_MainLayout.addWidget(new QLabel("Density Scale"), 2, 0);

	m_DensityScaleSlider.setOrientation(Qt::Horizontal);
	m_DensityScaleSlider.setRange(0.001, 100.0);
	m_DensityScaleSlider.setValue(scene->m_DensityScale);
	m_MainLayout.addWidget(&m_DensityScaleSlider, 2, 1);

	m_DensityScaleSpinner.setRange(0.001, 100.0);
	m_DensityScaleSpinner.setDecimals(3);
	m_DensityScaleSpinner.setValue(scene->m_DensityScale);
	m_MainLayout.addWidget(&m_DensityScaleSpinner, 2, 2);

	m_MainLayout.addWidget(new QLabel("Shading Type"), 3, 0);

	m_ShadingType.addItem("BRDF Only", 0);
	m_ShadingType.addItem("Phase Function Only", 1);
	m_ShadingType.addItem("Hybrid", 2);
	m_MainLayout.addWidget(&m_ShadingType, 3, 1, 1, 2);

	m_GradientFactorLabel.setText("Gradient Factor");
	m_MainLayout.addWidget(&m_GradientFactorLabel, 4, 0);
	
	m_GradientFactorSlider.setRange(0.001, 100.0);
	m_GradientFactorSlider.setValue(scene->m_GradientFactor);

	m_MainLayout.addWidget(&m_GradientFactorSlider, 4, 1);

	m_GradientFactorSpinner.setRange(0.001, 100.0);
	m_GradientFactorSpinner.setDecimals(3);
	m_GradientFactorSpinner.setValue(scene->m_GradientFactor);

	m_MainLayout.addWidget(&m_GradientFactorSpinner, 4, 2);

	QObject::connect(&m_DensityScaleSlider, SIGNAL(valueChanged(double)), &m_DensityScaleSpinner, SLOT(setValue(double)));
	QObject::connect(&m_DensityScaleSpinner, SIGNAL(valueChanged(double)), &m_DensityScaleSlider, SLOT(setValue(double)));
	QObject::connect(&m_DensityScaleSlider, SIGNAL(valueChanged(double)), this, SLOT(OnSetDensityScale(double)));

	QObject::connect(&m_GradientFactorSlider, SIGNAL(valueChanged(double)), &m_GradientFactorSpinner, SLOT(setValue(double)));
	QObject::connect(&m_GradientFactorSpinner, SIGNAL(valueChanged(double)), &m_GradientFactorSlider, SLOT(setValue(double)));
	QObject::connect(&m_GradientFactorSlider, SIGNAL(valueChanged(double)), this, SLOT(OnSetGradientFactor(double)));

	m_MainLayout.addWidget(new QLabel("Primary Step Size"), 5, 0);

	m_StepSizePrimaryRaySlider.setRange(1.0, 10.0);

	m_MainLayout.addWidget(&m_StepSizePrimaryRaySlider, 5, 1);

	m_StepSizePrimaryRaySpinner.setRange(1.0, 10.0);
	m_StepSizePrimaryRaySpinner.setDecimals(2);

	m_MainLayout.addWidget(&m_StepSizePrimaryRaySpinner, 5, 2);

	QObject::connect(&m_StepSizePrimaryRaySlider, SIGNAL(valueChanged(double)), &m_StepSizePrimaryRaySpinner, SLOT(setValue(double)));
	QObject::connect(&m_StepSizePrimaryRaySpinner, SIGNAL(valueChanged(double)), &m_StepSizePrimaryRaySlider, SLOT(setValue(double)));
	QObject::connect(&m_StepSizePrimaryRaySlider, SIGNAL(valueChanged(double)), this, SLOT(OnSetStepSizePrimaryRay(double)));

	m_MainLayout.addWidget(new QLabel("Secondary Step Size"), 6, 0);

	m_StepSizeSecondaryRaySlider.setRange(1.0, 10.0);

	m_MainLayout.addWidget(&m_StepSizeSecondaryRaySlider, 6, 1);

	m_StepSizeSecondaryRaySpinner.setRange(1.0, 10.0);
	m_StepSizeSecondaryRaySpinner.setDecimals(2);

	m_MainLayout.addWidget(&m_StepSizeSecondaryRaySpinner, 6, 2);

	QObject::connect(&m_StepSizeSecondaryRaySlider, SIGNAL(valueChanged(double)), &m_StepSizeSecondaryRaySpinner, SLOT(setValue(double)));
	QObject::connect(&m_StepSizeSecondaryRaySpinner, SIGNAL(valueChanged(double)), &m_StepSizeSecondaryRaySlider, SLOT(setValue(double)));
	QObject::connect(&m_StepSizeSecondaryRaySlider, SIGNAL(valueChanged(double)), this, SLOT(OnSetStepSizeSecondaryRay(double)));


	QObject::connect(&m_RendererType, SIGNAL(currentIndexChanged(int)), this, SLOT(OnSetRendererType(int)));
	QObject::connect(&m_ShadingType, SIGNAL(currentIndexChanged(int)), this, SLOT(OnSetShadingType(int)));
	//QObject::connect(&gStatus, SIGNAL(RenderBegin()), this, SLOT(OnRenderBegin()));
	
	QObject::connect(_transferFunction, SIGNAL(Changed()), this, SLOT(OnTransferFunctionChanged()));

}

void QAppearanceSettingsWidget::OnRenderBegin(void)
{
	m_DensityScaleSlider.setValue(_transferFunction->GetDensityScale());
	m_ShadingType.setCurrentIndex(_transferFunction->GetShadingType());
	m_GradientFactorSlider.setValue(_transferFunction->scene()->m_GradientFactor);

	m_StepSizePrimaryRaySlider.setValue(_transferFunction->scene()->m_StepSizeFactor, true);
	m_StepSizePrimaryRaySpinner.setValue(_transferFunction->scene()->m_StepSizeFactor, true);
	m_StepSizeSecondaryRaySlider.setValue(_transferFunction->scene()->m_StepSizeFactorShadow, true);
	m_StepSizeSecondaryRaySpinner.setValue(_transferFunction->scene()->m_StepSizeFactorShadow, true);
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
	m_GradientFactorSpinner.setEnabled(Index == 2);
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
	_transferFunction->scene()->m_StepSizeFactor = (float)StepSizePrimaryRay;
	_transferFunction->scene()->m_DirtyFlags.SetFlag(RenderParamsDirty);
}

void QAppearanceSettingsWidget::OnSetStepSizeSecondaryRay(const double& StepSizeSecondaryRay)
{
	_transferFunction->scene()->m_StepSizeFactorShadow = (float)StepSizeSecondaryRay;
	_transferFunction->scene()->m_DirtyFlags.SetFlag(RenderParamsDirty);
}

void QAppearanceSettingsWidget::OnTransferFunctionChanged(void)
{
	m_DensityScaleSlider.setValue(_transferFunction->GetDensityScale(), true);
	m_DensityScaleSpinner.setValue(_transferFunction->GetDensityScale(), true);
	m_ShadingType.setCurrentIndex(_transferFunction->GetShadingType());
	m_GradientFactorSlider.setValue(_transferFunction->GetGradientFactor(), true);
	m_GradientFactorSpinner.setValue(_transferFunction->GetGradientFactor(), true);
}

void QAppearanceSettingsWidget::OnDiffuseColorChanged(int i, const QColor& color)
{
	qreal rgba[4];
	color.getRgbF(&rgba[0], &rgba[1], &rgba[2], &rgba[3]);
	_scene->_material.diffuse[i * 3 + 0] = rgba[0];
	_scene->_material.diffuse[i * 3 + 1] = rgba[1];
	_scene->_material.diffuse[i * 3 + 2] = rgba[2];
	_transferFunction->scene()->m_DirtyFlags.SetFlag(RenderParamsDirty);
}

void QAppearanceSettingsWidget::OnSpecularColorChanged(int i, const QColor& color)
{
	qreal rgba[4];
	color.getRgbF(&rgba[0], &rgba[1], &rgba[2], &rgba[3]);
	_scene->_material.specular[i * 3 + 0] = rgba[0];
	_scene->_material.specular[i * 3 + 1] = rgba[1];
	_scene->_material.specular[i * 3 + 2] = rgba[2];
	_transferFunction->scene()->m_DirtyFlags.SetFlag(RenderParamsDirty);
}

void QAppearanceSettingsWidget::OnEmissiveColorChanged(int i, const QColor& color)
{
	qreal rgba[4];
	color.getRgbF(&rgba[0], &rgba[1], &rgba[2], &rgba[3]);
	_scene->_material.emissive[i * 3 + 0] = rgba[0];
	_scene->_material.emissive[i * 3 + 1] = rgba[1];
	_scene->_material.emissive[i * 3 + 2] = rgba[2];
	_transferFunction->scene()->m_DirtyFlags.SetFlag(RenderParamsDirty);
}
void QAppearanceSettingsWidget::OnSetWindowLevel(int i, double window, double level)
{
	_scene->_volume->channel((uint32_t)i)->generate_windowLevel(window, level);

	_transferFunction->scene()->m_DirtyFlags.SetFlag(TransferFunctionDirty);
}

inline QVector<QColor> rndColors(int count) {
	QVector<QColor> colors;
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

	// I don't own this.
	_scene = scene;

	const float DEFAULT_WINDOW = 1.0f;
	const float DEFAULT_LEVEL = 0.75f;
	QVector<QColor> colors = rndColors(scene->_volume->sizeC());

	for (uint32_t i = 0; i < scene->_volume->sizeC(); ++i) {
		Section* section = new Section(QString("Channel %1").arg(i), 0);

		auto* sectionLayout = new QGridLayout();

		int row = 0;
		sectionLayout->addWidget(new QLabel("Window"), row, 0);
		QDoubleSlider* windowSlider = new QDoubleSlider();
		windowSlider->setRange(0.001, 1.0);
		windowSlider->setValue(DEFAULT_WINDOW);
		sectionLayout->addWidget(windowSlider, row, 1);

		row++;
		sectionLayout->addWidget(new QLabel("Level"), row, 0);
		QDoubleSlider* levelSlider = new QDoubleSlider();
		levelSlider->setRange(0.001, 1.0);
		levelSlider->setValue(DEFAULT_LEVEL);
		sectionLayout->addWidget(levelSlider, row, 1);

		QObject::connect(windowSlider, &QDoubleSlider::valueChanged, [i, this, levelSlider](double d) {
			this->OnSetWindowLevel(i, d, levelSlider->value());
		});
		QObject::connect(levelSlider, &QDoubleSlider::valueChanged, [i, this, windowSlider](double d) {
			this->OnSetWindowLevel(i, windowSlider->value(), d);
		});
		// init
		this->OnSetWindowLevel(i, DEFAULT_WINDOW, DEFAULT_LEVEL);

		row++;
		QColorSelector* diffuseColorButton = new QColorSelector();
		diffuseColorButton->SetColor(colors[i], true);
		sectionLayout->addWidget(new QLabel("DiffuseColor"), row, 0);
		sectionLayout->addWidget(diffuseColorButton, row, 2);
		QObject::connect(diffuseColorButton, &QColorSelector::currentColorChanged, [i, this](const QColor& c) {
			this->OnDiffuseColorChanged(i, c);
		});
		// init
		this->OnDiffuseColorChanged(i, colors[i]);

		row++;
		QColorSelector* specularColorButton = new QColorSelector();
		specularColorButton->SetColor(QColor::fromRgbF(0.0f, 0.0f, 0.0f), true);
		sectionLayout->addWidget(new QLabel("SpecularColor"), row, 0);
		sectionLayout->addWidget(specularColorButton, row, 2);
		QObject::connect(specularColorButton, &QColorSelector::currentColorChanged, [i, this](const QColor& c) {
			this->OnSpecularColorChanged(i, c);
		});
		// init
		this->OnSpecularColorChanged(i, QColor::fromRgbF(0.0f, 0.0f, 0.0f));

		row++;
		QColorSelector* emissiveColorButton = new QColorSelector();
		emissiveColorButton->SetColor(QColor::fromRgbF(0.0f, 0.0f, 0.0f), true);
		sectionLayout->addWidget(new QLabel("EmissiveColor"), row, 0);
		sectionLayout->addWidget(emissiveColorButton, row, 2);
		QObject::connect(emissiveColorButton, &QColorSelector::currentColorChanged, [i, this](const QColor& c) {
			this->OnEmissiveColorChanged(i, c);
		});
		// init
		this->OnEmissiveColorChanged(i, QColor::fromRgbF(0.0f, 0.0f, 0.0f));

		section->setContentLayout(*sectionLayout);
		m_MainLayout.addWidget(section, 12+i, 0, 1, -1);
		_channelSections.push_back(section);
	}
}
