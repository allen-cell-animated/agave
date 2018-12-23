#pragma once

#include "DenoiseParams.cuh"
#include "Flags.h"

class RenderSettings
{
public:
  RenderSettings(void);
  RenderSettings(const RenderSettings& Other);
  RenderSettings& operator=(const RenderSettings& Other);

  CFlags m_DirtyFlags;
  CRenderSettings m_RenderSettings;
  CDenoiseParams m_DenoiseParams;

  int GetNoIterations(void) const { return m_NoIterations; }
  void SetNoIterations(const int& NoIterations) { m_NoIterations = NoIterations; }

private:
  int m_NoIterations;
};
