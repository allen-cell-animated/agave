#pragma once

#include "DenoiseParams.h"
#include "Flags.h"

class RenderSettings
{
public:
  RenderSettings(void);
  RenderSettings(const RenderSettings& Other);
  RenderSettings& operator=(const RenderSettings& Other);

  Flags m_DirtyFlags;
  PathTraceRenderSettings m_RenderSettings;
  DenoiseParams m_DenoiseParams;

  int GetNoIterations(void) const { return m_NoIterations; }
  void SetNoIterations(const int& NoIterations) { m_NoIterations = NoIterations; }

private:
  int m_NoIterations;
};
