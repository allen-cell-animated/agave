#pragma once

#include "Controls.h"

#include <QtWidgets/QComboBox>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QLabel>

class QTransferFunction;
class CScene;

class QAppearanceSettingsWidget : public QGroupBox
{
	Q_OBJECT

public:
    QAppearanceSettingsWidget(QWidget* pParent = NULL, QTransferFunction* tran = nullptr, CScene* scene = nullptr);

public slots:
	void OnRenderBegin(void);
	void OnSetDensityScale(double DensityScale);
	void OnTransferFunctionChanged(void);
	void OnSetRendererType(int Index);
	void OnSetShadingType(int Index);
	void OnSetGradientFactor(double GradientFactor);
	void OnSetStepSizePrimaryRay(const double& StepSizePrimaryRay);
	void OnSetStepSizeSecondaryRay(const double& StepSizeSecondaryRay);
	void OnDiffuseColorChanged(const QColor& color);
	void OnSpecularColorChanged(const QColor& color);
	void OnEmissiveColorChanged(const QColor& color);

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

	QColorSelector  m_DiffuseColorButton;
	QColorSelector  m_SpecularColorButton;
	QColorSelector  m_EmissiveColorButton;

	QTransferFunction* _transferFunction;
};