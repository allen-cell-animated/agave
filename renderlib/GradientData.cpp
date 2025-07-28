#include "GradientData.h"

#include "Histogram.h"

void
GradientData::convert(const Histogram& histogram, const Histogram& newHistogram)
{
  // pct can remain the same; percentiles are always relative to binned pixel counts?

  // min and max can also remain the same.

  // window/level:
  // 0 and 1 correspond to histogram._dataMin and histogram._dataMax
  float absoluteWindowSize = m_window * histogram.dataRange();
  float absoluteLevel = m_level * histogram.dataRange() + histogram._dataMin;

  m_window = absoluteWindowSize / newHistogram.dataRange();
  m_level = (absoluteLevel - newHistogram._dataMin) / newHistogram.dataRange();

  // convert Iso:
  float absoluteIsoRange = m_isorange * histogram.dataRange();
  float absoluteIsoValue = m_isovalue * histogram.dataRange() + histogram._dataMin;

  m_isorange = absoluteIsoRange / newHistogram.dataRange();
  m_isovalue = (absoluteIsoValue - newHistogram._dataMin) / newHistogram.dataRange();

  // convert "custom":
  for (int i = 0; i < m_customControlPoints.size(); ++i) {
    LutControlPoint p = m_customControlPoints[i];
    p.first = p.first * histogram.dataRange() + histogram._dataMin;
    p.first = (p.first - newHistogram._dataMin) / newHistogram.dataRange();
    m_customControlPoints[i] = p;
  }
}

bool
GradientData::getMinMax(const Histogram& histogram, uint16_t* imin, uint16_t* imax) const
{
  if (m_activeMode == GradientEditMode::WINDOW_LEVEL) {
    *imin = histogram._dataMin + m_level * histogram.dataRange() - m_window * histogram.dataRange() / 2;
    *imax = histogram._dataMin + m_level * histogram.dataRange() + m_window * histogram.dataRange() / 2;
    return true;
  } else if (m_activeMode == GradientEditMode::PERCENTILE) {
    float window, level;
    histogram.computeWindowLevelFromPercentiles(m_pctLow, m_pctHigh, window, level);
    *imin = histogram._dataMin + level * histogram.dataRange() - window * histogram.dataRange() / 2;
    *imax = histogram._dataMin + level * histogram.dataRange() + window * histogram.dataRange() / 2;
    return true;
  } else if (m_activeMode == GradientEditMode::ISOVALUE) {
    *imin = histogram._dataMin + m_isovalue * histogram.dataRange() - m_isorange * histogram.dataRange() / 2;
    *imax = histogram._dataMin + m_isovalue * histogram.dataRange() + m_isorange * histogram.dataRange() / 2;
    return true;
  } else if (m_activeMode == GradientEditMode::MINMAX) {
    *imin = m_minu16;
    *imax = m_maxu16;
    return true;
  } else {
    return false;
  }
}
