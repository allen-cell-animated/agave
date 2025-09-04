#include "RenderSettings.h"
#include "ImageXYZC.h"

RenderSettings::RenderSettings(void)
  : m_DirtyFlags()
  , m_DenoiseParams()
  , m_NoIterations(0)
  , m_rendererType(1)
{
}

RenderSettings::RenderSettings(const RenderSettings& Other)
{
  *this = Other;
}

RenderSettings&
RenderSettings::operator=(const RenderSettings& Other)
{
  m_DirtyFlags = Other.m_DirtyFlags;
  m_DenoiseParams = Other.m_DenoiseParams;
  m_NoIterations = Other.m_NoIterations;
  m_RenderSettings = Other.m_RenderSettings;
  m_rendererType = Other.m_rendererType;

  return *this;
}
