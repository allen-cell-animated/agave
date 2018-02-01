#pragma once

#include "Controls.h"

#include <QtWidgets/QComboBox>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLabel>

#include <memory>

class QTransferFunction;
class CScene;
class ImageXYZC;
class Scene;
class Section;

class QAppearanceSettingsWidget : public QGroupBox
{
	Q_OBJECT

public:
    QAppearanceSettingsWidget(QWidget* pParent = NULL, QTransferFunction* tran = nullptr, CScene* scene = nullptr);

	void onNewImage(Scene* scene);

public slots:
	void OnRenderBegin(void);
	void OnSetDensityScale(double DensityScale);
	void OnTransferFunctionChanged(void);
	void OnSetRendererType(int Index);
	void OnSetShadingType(int Index);
	void OnSetGradientFactor(double GradientFactor);
	void OnSetStepSizePrimaryRay(const double& StepSizePrimaryRay);
	void OnSetStepSizeSecondaryRay(const double& StepSizeSecondaryRay);

public:
	void OnDiffuseColorChanged(int i, const QColor& color);
	void OnSpecularColorChanged(int i, const QColor& color);
	void OnEmissiveColorChanged(int i, const QColor& color);
	void OnSetWindowLevel(int i, double window, double level);
	void OnRoughnessChanged(int i, double roughness);
	void OnChannelChecked(int i, bool is_checked);

	void OnSetAreaLightTheta(double value);
	void OnSetAreaLightPhi(double value);
	void OnSetAreaLightSize(double value);
	void OnSetAreaLightDistance(double value);
	void OnSetAreaLightColor(double intensity, const QColor& color);

private:
	QGridLayout		m_MainLayout;
	QDoubleSlider	m_DensityScaleSlider;
	QDoubleSpinner	m_DensityScaleSpinner;
	QComboBox		m_RendererType;
	QComboBox		m_ShadingType;
	QLabel			m_GradientFactorLabel;
	QDoubleSlider	m_GradientFactorSlider;
	QDoubleSpinner	m_GradientFactorSpinner;
	QDoubleSlider	m_StepSizePrimaryRaySlider;
	QDoubleSpinner	m_StepSizePrimaryRaySpinner;
	QDoubleSlider	m_StepSizeSecondaryRaySlider;
	QDoubleSpinner	m_StepSizeSecondaryRaySpinner;

	QTransferFunction* _transferFunction;

	Scene* _scene;
	std::vector<Section*> _channelSections;
};