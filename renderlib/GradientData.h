#pragma once

#include <inttypes.h>
#include <vector>

struct Histogram;

using LutControlPoint = std::pair<float, float>;

enum class GradientEditMode
{
  WINDOW_LEVEL,
  ISOVALUE,
  PERCENTILE,
  CUSTOM,
  MINMAX
};

struct GradientData
{
  GradientEditMode m_activeMode = GradientEditMode::PERCENTILE;

  float m_nonlinearityWL =
    0.0f; // 0 is linear, 1 is maximal upward curve exponent, -1 is maximal downward curve exponent
  float m_window = 0.25f;
  float m_level = 0.5f;

  float m_isovalue = 0.5f;
  float m_isorange = 0.1f;

  float m_pctLow = 0.5f;
  float m_pctHigh = 0.98f;
  float m_nonlinearityPct =
    0.0f; // 0 is linear, 1 is maximal upward curve exponent, -1 is maximal downward curve exponent

  uint16_t m_maxu16 = 65535;
  uint16_t m_minu16 = 0;
  float m_nonlinearityMinMax =
    0.0f; // 0 is linear, 1 is maximal upward curve exponent, -1 is maximal downward curve exponent

  std::vector<LutControlPoint> m_customControlPoints = { { 0.0f, 0.0f }, { 1.0f, 1.0f } };

  void convert(const Histogram& oldHistogram, const Histogram& newHistogram);

  // if the mode is such that a min/max exist (window/level or percentile), calculate them and return the histo
  // intensities at min and max
  bool getMinMax(const Histogram& histogram, uint16_t* imin, uint16_t* imax) const;
};
