#include "Stable.h"

#include "TransferFunction.h"

#include "Scene.h"

QTransferFunction::QTransferFunction(QObject* pParent) :
	QObject(pParent),
	m_DensityScale(100.0f),
	m_ShadingType(2),
	m_RendererType(1),
	m_GradientFactor(10.0f)
{
}

QTransferFunction::QTransferFunction(const QTransferFunction& Other)
{
	*this = Other;
};

QTransferFunction& QTransferFunction::operator = (const QTransferFunction& Other)			
{
	blockSignals(true);
	
	m_DensityScale		= Other.m_DensityScale;
	m_ShadingType		= Other.m_ShadingType;
	m_GradientFactor	= Other.m_GradientFactor;

	blockSignals(false);

	// Notify others that the function has changed selection has changed
	emit Changed();

	return *this;
}

void QTransferFunction::setScene(CScene& scene) { 
	_Scene = &scene;

	m_DensityScale = scene.m_DensityScale;
	m_ShadingType = scene.m_ShadingType;
	m_GradientFactor = scene.m_GradientFactor;

	emit Changed();
}

float QTransferFunction::GetDensityScale(void) const
{
	return m_DensityScale;
}

void QTransferFunction::SetDensityScale(const float& DensityScale)
{
	if (DensityScale == m_DensityScale)
		return;

	m_DensityScale = DensityScale;

	emit Changed();
}

int QTransferFunction::GetShadingType(void) const
{
	return m_ShadingType;
}

void QTransferFunction::SetShadingType(const int& ShadingType)
{
	if (ShadingType == m_ShadingType)
		return;

	m_ShadingType = ShadingType;

	emit Changed();
}

int QTransferFunction::GetRendererType(void) const
{
	return m_RendererType;
}

void QTransferFunction::SetRendererType(const int& RendererType)
{
	if (RendererType == m_RendererType)
		return;

	m_RendererType = RendererType;

	emit ChangedRenderer(RendererType);
}

float QTransferFunction::GetGradientFactor(void) const
{
	return m_GradientFactor;
}

void QTransferFunction::SetGradientFactor(const float& GradientFactor)
{
	if (GradientFactor == m_GradientFactor)
		return;

	m_GradientFactor = GradientFactor;

	emit Changed();
}

QTransferFunction QTransferFunction::Default(void)
{
	QTransferFunction DefaultTransferFunction;

	DefaultTransferFunction.SetDensityScale(100.0f);
	DefaultTransferFunction.SetShadingType(2);
	DefaultTransferFunction.SetGradientFactor(10.0f);
	
	return DefaultTransferFunction;
}