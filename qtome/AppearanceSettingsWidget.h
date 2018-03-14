#pragma once

#include "Controls.h"

#include <QtWidgets/QComboBox>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLabel>

#include <memory>

class QTransferFunction;
class ImageXYZC;
class RenderSettings;
class Scene;
class Section;

class QAppearanceSettingsWidget : public QGroupBox
{
	Q_OBJECT

public:
    QAppearanceSettingsWidget(QWidget* pParent = NULL, QTransferFunction* tran = nullptr, RenderSettings* rs = nullptr);

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
	void OnSetSkyLightTopColor(double intensity, const QColor& color);
	void OnSetSkyLightMidColor(double intensity, const QColor& color);
	void OnSetSkyLightBotColor(double intensity, const QColor& color);

	void OnSetRoiXMax(int value);
	void OnSetRoiYMax(int value);
	void OnSetRoiZMax(int value);
	void OnSetRoiXMin(int value);
	void OnSetRoiYMin(int value);
	void OnSetRoiZMin(int value);

private:
	QGridLayout		m_MainLayout;
	QNumericSlider m_DensityScaleSlider;
	QComboBox		m_RendererType;
	QComboBox		m_ShadingType;
	QLabel			m_GradientFactorLabel;
	QNumericSlider	m_GradientFactorSlider;
	QNumericSlider	m_StepSizePrimaryRaySlider;
	QNumericSlider	m_StepSizeSecondaryRaySlider;

	QTransferFunction* _transferFunction;

	Section* _clipRoiSection;

	Scene* _scene;
	std::vector<Section*> _channelSections;
};