#pragma once

#include <algorithm>
#include <inttypes.h>

class Timeline
{
  int32_t m_MinTime = 0;
  int32_t m_MaxTime = 0;
  int32_t m_RangeSize = 1;
  int32_t m_CurrentTime = 0;

  // false means clamp, true means wrap
  bool m_wrap = true;

  int32_t forceInRange(int32_t t);

public:
  Timeline();
  Timeline(int32_t minTime, int32_t maxTime, bool wrapMode = true);

  int32_t currentTime() const { return m_CurrentTime; }

  int32_t increment(int32_t delta);

  int32_t setCurrentTime(int32_t t);

  void setRange(int32_t minT, int32_t maxT);
};
