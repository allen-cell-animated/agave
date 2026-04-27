#pragma once

#include <algorithm>
#include <inttypes.h>
#include <string>

class Timeline
{
public:
  enum class WrapMode
  {
    TIMELINE_WRAP,
    TIMELINE_CLAMP
  };
  Timeline();
  Timeline(int32_t minTime, int32_t maxTime, WrapMode wrapMode = WrapMode::TIMELINE_WRAP);

  int32_t currentTime() const { return m_CurrentTime; }
  int32_t minTime() const { return m_MinTime; }
  int32_t maxTime() const { return m_MaxTime; }
  double timeUnit() const { return m_TimeUnit; }
  const std::string& timeUnits() const { return m_TimeUnits; }

  int32_t increment(int32_t delta);

  int32_t setCurrentTime(int32_t t);

  void setRange(int32_t minT, int32_t maxT);

  void setTimeUnit(double timeUnit);
  void setTimeUnits(const std::string& timeUnits) { m_TimeUnits = timeUnits; }
  double toPhysicalTime(int32_t t) const;
  double currentPhysicalTime() const { return toPhysicalTime(m_CurrentTime); }

  void setWrap(WrapMode wrapMode) { m_wrapMode = wrapMode; }

private:
  int32_t m_MinTime = 0;
  int32_t m_MaxTime = 0;
  int32_t m_RangeSize = 1;
  int32_t m_CurrentTime = 0;
  double m_TimeUnit = 1.0;
  std::string m_TimeUnits = "s";

  // false means clamp, true means wrap
  WrapMode m_wrapMode = WrapMode::TIMELINE_WRAP;

  int32_t forceInRange(int32_t t);
};
