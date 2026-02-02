#pragma once

#include "GradientData.h"

#include <inttypes.h>
#include <stddef.h>
#include <vector>

struct Histogram
{
  Histogram(uint16_t* data, size_t length, size_t bins = 512);

  static constexpr float DEFAULT_PCT_LOW = 0.5f;
  static constexpr float DEFAULT_PCT_HIGH = 0.983f;

  // no more than 2^32 pixels of any one intensity in the data!?!?!
  std::vector<uint32_t> _bins;
  // cumulative counts from low to high
  std::vector<uint32_t> _ccounts;
  uint16_t _dataMin;
  uint16_t _dataMax;
  size_t _dataMinIdx;
  size_t _dataMaxIdx;
  // index of bin with most pixels
  size_t _maxBin;
  size_t _pixelCount;

  void computeWindowLevelFromPercentiles(float pct_low, float pct_high, float& window, float& level) const;

  float* generate_fullRange(size_t length = 256) const;
  float* generate_dataRange(size_t length = 256) const;
  float* generate_bestFit(size_t length = 256) const;
  // attempt to redo imagej's Auto
  float* generate_auto2(size_t length = 256) const;
  float* generate_auto(size_t length = 256) const;
  float* generate_percentiles(float lo = DEFAULT_PCT_LOW, float hi = DEFAULT_PCT_HIGH, size_t length = 256) const;
  float* generate_windowLevel(float window, float level, size_t length = 256) const;
  float* generate_controlPoints(std::vector<LutControlPoint> pts, size_t length = 256) const;
  float* generate_equalized(size_t length = 256) const;

  uint16_t dataRange() const { return _dataMax - _dataMin; }

  // Determine center values for first and last bins, and bin size.
  void bin_range(uint32_t nbins, float& firstBinCenter, float& lastBinCenter, float& binSize) const;
  std::vector<uint32_t> bin_counts(uint32_t nbins);
  float rank_data_value(float fraction) const;
  float* initialize_thresholds(float vfrac_min = 0.01f, float vfrac_max = 0.90f) const;

  float* generateFromGradientData(const GradientData& gradientData, size_t length = 256) const;

  size_t getBinOfIntensity(uint16_t intensity) const;
  void computePercentile(uint16_t intensity, float& percentile) const;
};
