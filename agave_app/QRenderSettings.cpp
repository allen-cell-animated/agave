#include "QRenderSettings.h"

#include "RenderSettings.h"

QRenderSettings::QRenderSettings(QObject* pParent)
  : QObject(pParent)
  , m_renderSettings(nullptr)
{
}

QRenderSettings::QRenderSettings(const QRenderSettings& Other)
{
  *this = Other;
};

QRenderSettings&
QRenderSettings::operator=(const QRenderSettings& Other)
{
  blockSignals(true);

  *m_renderSettings = *Other.m_renderSettings;

  blockSignals(false);

  // Notify others that the function has changed selection has changed
  emit Changed();

  return *this;
}

void
QRenderSettings::setRenderSettings(RenderSettings& rs)
{
  m_renderSettings = &rs;

  emit Changed();
}

float
QRenderSettings::GetDensityScale(void) const
{
  return m_renderSettings->m_RenderSettings.m_DensityScale;
}

void
QRenderSettings::SetDensityScale(const float& DensityScale)
{
  if (DensityScale == m_renderSettings->m_RenderSettings.m_DensityScale)
    return;

  m_renderSettings->m_RenderSettings.m_DensityScale = DensityScale;

  emit Changed();
}

int
QRenderSettings::GetShadingType(void) const
{
  return m_renderSettings->m_RenderSettings.m_ShadingType;
}

void
QRenderSettings::SetShadingType(const int& ShadingType)
{
  if (ShadingType == m_renderSettings->m_RenderSettings.m_ShadingType)
    return;

  m_renderSettings->m_RenderSettings.m_ShadingType = ShadingType;

  emit Changed();
}

int
QRenderSettings::GetRendererType(void) const
{
  return m_renderSettings->m_rendererType;
}

void
QRenderSettings::SetRendererType(const int& RendererType)
{
  if (RendererType == m_renderSettings->m_rendererType)
    return;

  m_renderSettings->m_rendererType = RendererType;

  emit ChangedRenderer(RendererType);
}

float
QRenderSettings::GetGradientFactor(void) const
{
  return m_renderSettings->m_RenderSettings.m_GradientFactor;
}

void
QRenderSettings::SetGradientFactor(const float& GradientFactor)
{
  if (GradientFactor == m_renderSettings->m_RenderSettings.m_GradientFactor)
    return;

  m_renderSettings->m_RenderSettings.m_GradientFactor = GradientFactor;

  emit Changed();
}
