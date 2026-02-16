#include "Timeline.h"

int32_t
Timeline::forceInRange(int32_t t)
{
  if (m_wrapMode == WrapMode::TIMELINE_WRAP) {
    // if current time is below min, then advance it until it is in range or above!
    if (t < m_MinTime) {
      t += m_RangeSize * ((m_MinTime - t) / m_RangeSize + 1);
    }
    // wrap!
    t = m_MinTime + (t - m_MinTime) % m_RangeSize;
  } else {
    // clamp
    t = std::max(std::min(t, m_MaxTime), m_MinTime);
  }
  return t;
}

Timeline::Timeline()
  : Timeline(0, 0)
{
}

Timeline::Timeline(int32_t minTime, int32_t maxTime, WrapMode wrapMode)
  : m_MinTime(minTime)
  , m_MaxTime(maxTime)
  , m_CurrentTime(minTime)
  , m_RangeSize(maxTime - minTime + 1)
  , m_wrapMode(wrapMode)
{
}

int32_t
Timeline::increment(int32_t delta)
{
  m_CurrentTime = forceInRange(m_CurrentTime + delta);
  return m_CurrentTime;
}

int32_t
Timeline::setCurrentTime(int32_t t)
{
  m_CurrentTime = forceInRange(t);
  return m_CurrentTime;
}

void
Timeline::setRange(int32_t minT, int32_t maxT)
{
  m_MinTime = minT;
  m_MaxTime = maxT;
  m_RangeSize = m_MaxTime - m_MinTime + 1;

  m_CurrentTime = forceInRange(m_CurrentTime);
}

void
Timeline::setTimeUnit(double timeUnit)
{
  m_TimeUnit = (timeUnit > 0.0) ? timeUnit : 1.0;
}

double
Timeline::toPhysicalTime(int32_t t) const
{
  return static_cast<double>(t) * m_TimeUnit;
}
