#pragma once

#include <vector>
struct Histogram;

using LutControlPoint = std::pair<float, float>;

enum class GradientEditMode
{
  WINDOW_LEVEL,
  ISOVALUE,
  PERCENTILE,
  CUSTOM
};

struct GradientData
{
  GradientEditMode m_activeMode = GradientEditMode::PERCENTILE;
  float m_window = 0.25f;
  float m_level = 0.5f;
  float m_isovalue = 0.5f;
  float m_isorange = 0.1f;
  float m_pctLow = 0.5f;
  float m_pctHigh = 0.98f;
  std::vector<LutControlPoint> m_customControlPoints = { { 0.0f, 0.0f }, { 1.0f, 1.0f } };

  void convert(const Histogram& oldHistogram, const Histogram& newHistogram);
};
