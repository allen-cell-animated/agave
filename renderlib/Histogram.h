#pragma once

#include "GradientData.h"

#include <inttypes.h>
#include <stddef.h>
#include <vector>

struct Histogram
{
  Histogram(uint16_t* data, size_t length);

  static constexpr float DEFAULT_PCT_LOW = 0.5f;
  static constexpr float DEFAULT_PCT_HIGH = 0.983f;

private:
  // no more than 2^32 pixels of any one intensity in the data!?!?!
  std::vector<uint32_t> _bins;         // more bins and smaller bin size, gives more accurate percentile computations
  std::vector<uint32_t> _filteredBins; // filtered bins for display

  // cumulative counts from low to high
  std::vector<uint32_t> _ccounts;
  uint16_t _dataMin;
  uint16_t _dataMax;
  size_t _dataMinIdx;
  size_t _dataMaxIdx;
  uint16_t _filteredMin;
  uint16_t _filteredMax;
  // index of bin with most pixels
  size_t _maxFilteredBin;
  size_t _pixelCount;

public:
  // return actual true absolute data extrema
  uint16_t getDataMin() const { return _dataMin; }
  uint16_t getDataMax() const { return _dataMax; }
  size_t getDataMinIdx() const { return _dataMinIdx; }
  size_t getDataMaxIdx() const { return _dataMaxIdx; }

  // outlier-filtered data extrema
  uint16_t getFilteredMin() const { return _filteredMin; }
  uint16_t getFilteredMax() const { return _filteredMax; }

  size_t getPixelCount() const { return _pixelCount; }

  // get the number of pixels in a display bin
  size_t getDisplayBinCount(size_t bin) const;
  size_t getModalDisplayBin() const { return _maxFilteredBin; }
  size_t getNumDisplayBins() const { return _filteredBins.size(); }

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

  // Given a number of bins, determine center values for first and last bins,
  // and bin size, based on the outlier-filtered min and max data values.
  void filteredBinRange(uint32_t nbins, float& firstBinCenter, float& lastBinCenter, float& binSize) const;
  float rank_data_value(float fraction) const;
  float* initialize_thresholds(float vfrac_min = 0.01f, float vfrac_max = 0.90f) const;

  float* generateFromGradientData(const GradientData& gradientData, size_t length = 256) const;

  size_t getBinOfIntensity(uint16_t intensity) const;
  void computePercentile(uint16_t intensity, float& percentile) const;
};
