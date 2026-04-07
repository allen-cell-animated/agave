#include "GradientData.h"

#include "Histogram.h"
#include "Logging.h"

void
GradientData::convert(const Histogram& histogram, const Histogram& newHistogram)
{
  // pct can remain the same; percentiles are always relative to binned pixel counts?

  // min and max can also remain the same.

  // window/level:
  // 0 and 1 correspond to histogram._dataMin and histogram._dataMax
  float absoluteWindowSize = m_window * histogram.dataRange();
  float absoluteLevel = m_level * histogram.dataRange() + histogram.getDataMin();

  m_window = absoluteWindowSize / newHistogram.dataRange();
  m_level = (absoluteLevel - newHistogram.getDataMin()) / newHistogram.dataRange();

  // convert Iso:
  float absoluteIsoRange = m_isorange * histogram.dataRange();
  float absoluteIsoValue = m_isovalue * histogram.dataRange() + histogram.getDataMin();
  m_isorange = absoluteIsoRange / newHistogram.dataRange();
  m_isovalue = (absoluteIsoValue - newHistogram.getDataMin()) / newHistogram.dataRange();

  // convert "custom":
  for (int i = 0; i < m_customControlPoints.size(); ++i) {
    LutControlPoint p = m_customControlPoints[i];
    uint16_t intensity = histogram.getDataMin() + static_cast<uint16_t>(p.first * histogram.dataRange());
    p.first = (float)(intensity - newHistogram.getDataMin()) / (float)newHistogram.dataRange();
    m_customControlPoints[i] = p;
  }
  std::sort(m_customControlPoints.begin(),
            m_customControlPoints.end(),
            [](const LutControlPoint& a, const LutControlPoint& b) { return a.first < b.first; });
}

bool
GradientData::getMinMax(const Histogram& histogram, uint16_t* imin, uint16_t* imax) const
{
  if (m_activeMode == GradientEditMode::WINDOW_LEVEL) {
    *imin = histogram.getDataMin() + m_level * histogram.dataRange() - m_window * histogram.dataRange() / 2;
    *imax = histogram.getDataMin() + m_level * histogram.dataRange() + m_window * histogram.dataRange() / 2;
    return true;
  } else if (m_activeMode == GradientEditMode::PERCENTILE) {
    float window, level;
    histogram.computeWindowLevelFromPercentiles(m_pctLow, m_pctHigh, window, level);
    *imin = histogram.getDataMin() + level * histogram.dataRange() - window * histogram.dataRange() / 2;
    *imax = histogram.getDataMin() + level * histogram.dataRange() + window * histogram.dataRange() / 2;
    return true;
  } else if (m_activeMode == GradientEditMode::ISOVALUE) {
    *imin = histogram.getDataMin() + m_isovalue * histogram.dataRange() - m_isorange * histogram.dataRange() / 2;
    *imax = histogram.getDataMin() + m_isovalue * histogram.dataRange() + m_isorange * histogram.dataRange() / 2;
    return true;
  } else if (m_activeMode == GradientEditMode::MINMAX) {
    *imin = m_minu16;
    *imax = m_maxu16;
    return true;
  } else {
    return false;
  }
}

std::vector<LutControlPoint>
GradientData::getControlPoints(const Histogram& histogram) const
{
  static constexpr float EPSILON = 0.00001f;
  uint16_t imin, imax;
  if (getMinMax(histogram, &imin, &imax)) {
    float fmin, fmax;
    fmin = (float)(imin - histogram.getDataMin()) / histogram.dataRange();
    fmax = (float)(imax - histogram.getDataMin()) / histogram.dataRange();
    // allow for fmin and fmax to be outside 0-1 range
    // also note that the ISO mode graph needs to be a step function.
    if (m_activeMode == GradientEditMode::ISOVALUE) {
      std::vector<LutControlPoint> pts = { { std::min(fmin - EPSILON - EPSILON, 0.0f), 0.0f },
                                           { fmin - EPSILON, 0.0f },
                                           { fmin + EPSILON, 1.0f },
                                           { fmax - EPSILON, 1.0f },
                                           { fmax + EPSILON, 0.0f },
                                           { std::max(fmax + EPSILON + EPSILON, 1.0f), 0.0f } };
      return pts;
    }
    std::vector<LutControlPoint> pts = {
      { std::min(fmin - EPSILON, 0.0f), 0.0f }, { fmin, 0.0f }, { fmax, 1.0f }, { std::max(fmax + EPSILON, 1.0f), 1.0f }
    };
    return pts;
  } else {
    return m_customControlPoints;
  }
}

bool
GradientData::getMinMax(const Histogram& histogram, std::pair<float, float>* minMax) const
{
  float fmin, fmax;
  float dataMin = static_cast<float>(histogram.getDataMin());
  float dataMax = static_cast<float>(histogram.getDataMax());
  float dataRange = dataMax - dataMin;
  if (dataRange <= 0.0f) {
    LOG_ERROR << "Data range is zero or negative in getMinMax: " << dataRange;
    return false;
  }

  if (m_activeMode == GradientEditMode::MINMAX) {
    fmin = static_cast<float>(m_minu16);
    fmax = static_cast<float>(m_maxu16);
  } else if (m_activeMode == GradientEditMode::WINDOW_LEVEL) {
    float lowEnd = m_level - m_window * 0.5f;
    float highEnd = m_level + m_window * 0.5f;
    lowEnd = std::max(0.0f, lowEnd);
    highEnd = std::min(1.0f, highEnd);
    fmin = dataMin + lowEnd * dataRange;
    fmax = dataMin + highEnd * dataRange;
  } else if (m_activeMode == GradientEditMode::PERCENTILE) {
    fmin = histogram.rank_data_value(m_pctLow);
    fmax = histogram.rank_data_value(m_pctHigh);
  } else if (m_activeMode == GradientEditMode::ISOVALUE) {
    float lowEnd = m_isovalue - m_isorange * 0.5f;
    float highEnd = m_isovalue + m_isorange * 0.5f;
    lowEnd = std::max(0.0f, lowEnd);
    highEnd = std::min(1.0f, highEnd);
    fmin = dataMin + lowEnd * dataRange;
    fmax = dataMin + highEnd * dataRange;
  } else {
    return false;
  }

  fmin = std::max(fmin, dataMin);
  fmax = std::min(fmax, dataMax);
  *minMax = { fmin, fmax };
  return true;
}