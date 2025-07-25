#include "Histogram.h"

#include "GradientData.h"
#include "Logging.h"
#include "MathUtil.h"

#include <algorithm>
#include <math.h>
#include <numeric>

template<class T>
const T&
clamp(const T& v, const T& lo, const T& hi)
{
  assert(hi > lo);
  return (v < lo) ? lo : (v > hi ? hi : v);
}

const float Histogram::DEFAULT_PCT_LOW = 0.5f;
const float Histogram::DEFAULT_PCT_HIGH = 0.983f;

Histogram::Histogram(uint16_t* data, size_t length, size_t num_bins)
  : _bins(num_bins)
  , _ccounts(num_bins)
  , _dataMin(0)
  , _dataMax(0)
{
  std::fill(_bins.begin(), _bins.end(), 0);

  if (data) {
    _dataMin = data[0];
    _dataMax = data[0];
  }

  uint16_t val;
  for (size_t i = 0; i < length; ++i) {
    val = data[i];
    if (val > _dataMax) {
      _dataMax = val;
    } else if (val < _dataMin) {
      _dataMin = val;
    }
  }
  //	float fval;
  float rangeMin = (float)_dataMin;
  float rangeMax = (float)_dataMax;
  float range = (float)(rangeMax - rangeMin);
  if (range == 0.0f) {
    range = 1.0f;
  }
  float binmax = (float)(num_bins - 1);
  for (size_t i = 0; i < length; ++i) {
    size_t whichbin = (size_t)((float)(data[i] - rangeMin) / range * binmax + 0.5);
    //		val = data[i];
    //		// normalize to 0..1 range
    //		// ZERO BIN is _dataMin intensity!!!!!! _dataMin MIGHT be nonzero.
    //		fval = (float)(val - _dataMin) / range;
    //		// select a bin
    //		fval *= binmax;
    //		// discretize (drop the fractional part?)
    //		size_t whichbin = (size_t)fval;
    _bins[whichbin]++;
    // bins goes from min to max of data range. not datatype range.
  }

  // total number of pixels
  _pixelCount = length;

  // get the bin with the most frequently occurring value
  _maxBin = 0;
  uint32_t curmax = _bins[0];
  for (size_t i = 1; i < _bins.size(); i++) {
    if (_bins[i] > curmax) {
      _maxBin = i;
      curmax = _bins[i];
    }
  }

  // add cumulative counts into the ccounts array
  std::partial_sum(_bins.begin(), _bins.end(), _ccounts.begin(), std::plus<uint32_t>());
  // last ccount bin should have total number of intensities.
  assert(_pixelCount == _ccounts[_ccounts.size() - 1]);
}

float*
Histogram::generate_fullRange(size_t length) const
{
  float window = 1.0;
  float level = 0.5;
  return generate_windowLevel(window, level, length);
}

float*
Histogram::generate_dataRange(size_t length) const
{
  float window = 1.0;
  float level = 0.5;
  return generate_windowLevel(window, level, length);
}

float*
Histogram::generate_bestFit(size_t length) const
{
  size_t pixcount = _pixelCount;
  size_t limit = pixcount / 10;

  size_t i = 0;
  size_t count = 0;
  for (i = 0; i < _bins.size(); ++i) {
    count += _bins[i];
    if (count > limit) {
      break;
    }
  }
  size_t hmin = i;

  count = 0;
  for (i = _bins.size() - 1; i >= 0; --i) {
    count += _bins[i];
    if (count > limit) {
      break;
    }
  }
  size_t hmax = i;

  size_t range = hmax - hmin;
  if (range < 1) {
    range = _bins.size() - 1;
  }

  float window = (float)(range) / (float)(_bins.size() - 1);
  float level = ((float)hmin + (float)range * 0.5f) / (float)(_bins.size() - 1);
  return generate_windowLevel(window, level, length);
}

// attempt to redo imagej's Auto
float*
Histogram::generate_auto2(size_t length) const
{

  size_t AUTO_THRESHOLD = 10000;
  size_t pixcount = _pixelCount; // -_bins[0];
  //  const pixcount = this.imgData.data.length;
  size_t limit = pixcount / 10;
  size_t threshold = pixcount / AUTO_THRESHOLD;

  size_t nbins = _bins.size();

#if 0
	size_t i = -1;
	bool found = false;
	int count;
	do {
		i++;
		count = _bins[i];
		if (count>limit) count = 0;
		found = count> threshold;
	} while (!found && i<(nbins-1));
	size_t hmin = i;
	i = nbins;
	do {
		i--;
		count = _bins[i];
		if (count>limit) count = 0;
		found = count > threshold;
	} while (!found && i > 0);
	size_t hmax = i;
#endif

  // #if 0
  //  this will skip the "zero" bin which contains pixels of zero intensity.
  size_t hmin = nbins - 1;
  size_t hmax = 0;
  for (size_t i = 0; i < nbins; ++i) {
    if (_bins[i] > threshold && _bins[i] <= limit) {
      hmin = i;
      break;
    }
  }
  // need a signed loop control variable here to test for >= 0.
  for (int i = (int)nbins - 1; i >= 0; --i) {
    if (_bins[i] > threshold && _bins[i] <= limit) {
      hmax = i;
      break;
    }
  }
  // #endif

  if (hmax < hmin) {
    // just reset to whole range in this case.
    return generate_fullRange(length);
  } else {
    // LOG_DEBUG << "auto2 range: " << hmin << "..." << hmax;
    float range = (float)hmax - (float)hmin;
    float window = (range) / (float)(nbins - 1);
    float level = ((float)hmin + range * 0.5f) / (float)(nbins - 1);
    // LOG_DEBUG << "auto2 window/level: " << window << " / " << level;
    return generate_windowLevel(window, level, length);
  }
}

float*
Histogram::generate_auto(size_t length) const
{

  // simple linear mapping cutting elements with small appearance
  // get 10% threshold
  float PERCENTAGE = 0.1f;
  float th = std::floor(_bins[_maxBin] * PERCENTAGE);
  size_t b = 0;
  size_t e = _bins.size() - 1;
  for (size_t x = 0; x < _bins.size(); ++x) {
    if (_bins[x] > th) {
      b = x;
      break;
    }
  }
  for (size_t x = _bins.size() - 1; x >= 0; --x) {
    if (_bins[x] > th) {
      e = x;
      break;
    }
  }

  size_t range = e - b;
  if (range < 1) {
    range = _bins.size() - 1;
    b = 0;
  }
  // if range == e-b, then
  // b+range/2 === e-range/2
  //
  float window = (float)range / (float)(_bins.size() - 1);
  float level = ((float)b + (float)range * 0.5f) / (float)(_bins.size() - 1);
  return generate_windowLevel(window, level, length);
}

/**
 * Generate a piecewise linear lookup table that ramps up from 0 to 1 over the b to e domain
 *  |
 * 1|               +---------+-----
 *  |              /
 *  |             /
 *  |            /
 *  |           /
 *  |          /
 * 0+=========+---------------+-----
 *  0         b    e          1
 * window = e-b      width of range e,b
 * level = 0.5*(e+b) midpoint of e,b
 */
// window and level are percentages of full range 0..1
float*
Histogram::generate_windowLevel(float window, float level, size_t length) const
{
  // return a LUT with new values(?)
  // data type of lut values is out_phys_range (uint8)
  // length of lut is number of histogram bins (represents the input data range)
  float* lut = new float[length];

  float a = (level - window * 0.5f);
  float b = (level + window * 0.5f);
  // b-a should be equal to window!
  assert(fabs(b - a - window) < 0.0001);
  float range = window;
  // float range = b - a;

  for (size_t x = 0; x < length; ++x) {
    float v = ((float)x / (float)(length - 1) - a) / range;
    lut[x] = clamp(v, 0.0f, 1.0f);
  }

  return lut;
}

void
Histogram::computeWindowLevelFromPercentiles(float pct_low, float pct_high, float& window, float& level) const
{
  // e.g. 0.50, 0.983 starts from 50th percentile bucket and ends at 98.3 percentile bucket.
  if (pct_low > pct_high) {
    std::swap(pct_high, pct_low);
  }

  size_t length = _bins.size();

  size_t lowlimit = size_t(_pixelCount * pct_low);
  size_t hilimit = size_t(_pixelCount * pct_high);

  // TODO use _ccounts in these loops!!

  size_t i = 0;
  size_t count = 0;
  for (i = 0; i < _bins.size(); ++i) {
    count += _bins[i];
    if (count > lowlimit) {
      break;
    }
  }
  size_t hmin = i;

  count = 0;
  for (i = 0; i < _bins.size(); ++i) {
    count += _bins[i];
    if (count > hilimit) {
      break;
    }
  }
  size_t hmax = i;

  // calculate a window and level that are percentages of the full range (0 .. length-1)
  window = (float)(hmax - hmin) / (float)(length - 1);
  level = (float)(hmin + hmax) * 0.5f / (float)(length - 1);
}

float*
Histogram::generate_percentiles(float lo, float hi, size_t length) const
{
  float window, level;
  computeWindowLevelFromPercentiles(lo, hi, window, level);
  return generate_windowLevel(window, level, length);
}

float*
Histogram::generate_controlPoints(std::vector<LutControlPoint> pts, size_t length) const
{
  // pts is piecewise linear from first to last control point.
  // pts is in order of increasing x value (the first element of the pair)
  // pts[0].first === 0
  // pts[pts.size()-1].first === 1

  float* lut = new float[length]{ 0 };

  for (size_t x = 0; x < length; ++x) {
    float fx = (float)x / (float)(length - 1);
    // find the interval of control points that contains fx.
    for (size_t i = 0; i < pts.size() - 1; ++i) {
      // am i in between?
      if ((fx >= pts[i].first) && (fx <= pts[i + 1].first)) {
        // what fraction of this interval in x?
        float fxi = (fx - pts[i].first) / (pts[i + 1].first - pts[i].first);
        // use that fraction against y range
        lut[x] = pts[i].second + fxi * (pts[i + 1].second - pts[i].second);
        break;
      }
    }
  }
  return lut;
}

void
Histogram::bin_range(uint32_t nbins, float& firstBinCenter, float& lastBinCenter, float& binSize) const
{
  uint16_t dmin = _dataMin;
  uint16_t dmax = _dataMax;
  float fbc, lbc, bsize;
  if (nbins > 1) {
    if (dmax > dmin) {
      fbc = dmin;
      lbc = dmax;
    } else {
      fbc = dmin - 1.0f;
      lbc = dmax + 1.0f;
    }
    bsize = (lbc - fbc) / (nbins - 1);
  } else {
    fbc = lbc = 0.5f * (dmax + dmin);
    bsize = 2.0f * (dmax - dmin);
    if (bsize <= 0) {
      bsize = 2;
    }
  }
  firstBinCenter = fbc;
  lastBinCenter = lbc;
  binSize = bsize;
}

// Compute histogram from binned data using a different number of bins.
std::vector<uint32_t>
Histogram::bin_counts(uint32_t nbins)
{
  uint32_t fbins = (uint32_t)_bins.size();
  std::vector<uint32_t>& fcounts = _bins;

  float fbc, lbc, bsize;
  bin_range(nbins, fbc, lbc, bsize);
  float ffbc, flbc, fbsize;
  bin_range(fbins, ffbc, flbc, fbsize);
  float r = bsize / fbsize;
  float s = 0.5f + ((fbc - ffbc) / fbsize);

  std::vector<uint32_t> bcounts(nbins, 0);
  for (uint32_t b = 0; b < nbins; ++b) {
    float fb0 = s + (b - 0.5f) * r;
    int b0 = int(ceil(fb0));
    float f0 = b0 - fb0;
    if (b0 < 0) {
      b0 = 0;
      f0 = 0;
    }
    float fb1 = s + (b + 0.5f) * r;
    int b1 = int(floor(fb1));
    float f1 = fb1 - b1;
    if (b1 >= (int32_t)fbins) {
      b1 = fbins;
      f1 = 0;
    }
    uint32_t c = 0;
    if ((b0 - 1) == b1) {
      c += (uint32_t)(r * fcounts[b0 - 1]);
    } else {
      if ((b0 > 0) && (b0 <= (int32_t)fbins)) {
        c += (uint32_t)(fcounts[b0 - 1] * f0);
      }
      if (b1 > b0) {
        // c += sum(fcounts[b0:b1]);
        for (int j = b0; j < b1; ++j) {
          c += fcounts[j];
        }
      }
      if ((b1 >= 0) && (b1 < (int32_t)fbins)) {
        c += (uint32_t)(fcounts[b1] * f1);
      }
    }
    bcounts[b] = c;
  }
  return bcounts;
}

// Find the data value where a specified fraction of voxels have lower value.
// Result is an approximation using binned data.
float
Histogram::rank_data_value(float fraction) const
{
  float targetcount = fraction * _ccounts[_ccounts.size() - 1];
  int b = 0;
  // assumes ccounts is monotonically increasing.
  // b is the index where one would insert targetcount in ccounts.
  for (; b < _ccounts.size(); ++b) {
    if (_ccounts[b] > targetcount) {
      break;
    }
  }
  // int b = _ccounts.searchsorted(fraction*_ccounts[_ccounts.size()-1]);
  float fbc, lbc, bsize;
  bin_range((uint32_t)_bins.size(), fbc, lbc, bsize);
  float v = fbc + b * (lbc - fbc) / (float)_bins.size();
  return v;
}

float*
Histogram::initialize_thresholds(float vfrac_min /*= 0.01*/, float vfrac_max /*= 0.90*/) const
{
  float ilow = 0.0f;
  float imid = 0.8f;
  float imax = 1.0f;
  float vlow = rank_data_value(1.0f - vfrac_max);
  float vmid = rank_data_value(1.0f - vfrac_min);
  float vmax = _dataMax;
  // LOG_DEBUG << "LOW: " << vlow << " HIGH: " << vmid;

  // normalize to 0..1
  float range = (float)(_dataMax - _dataMin);
  if (range == 0.0f) {
    range = 1.0f;
  }
  vlow = (vlow - _dataMin) / range;
  vmid = (vmid - _dataMin) / range;
  vmax = 1.0f;

  if ((vlow < vmid) && (vmid < vmax)) {
    if (vlow == 0.0f) {
      return generate_controlPoints({ { vlow, ilow }, { vmid, imid }, { vmax, imax } });
    } else {
      return generate_controlPoints({ { 0.0f, 0.0f }, { vlow, ilow }, { vmid, imid }, { vmax, imax } });
    }
  } else {
    if (vlow == 0.0f) {
      return generate_controlPoints({ { vlow, ilow }, { 0.9f * vlow + 0.1f * vmax, imid }, { vmax, imax } });
    } else {
      return generate_controlPoints(
        { { 0.0f, 0.0f }, { vlow, ilow }, { 0.9f * vlow + 0.1f * vmax, imid }, { vmax, imax } });
    }
  }
}

float*
Histogram::generate_equalized(size_t length) const
{
  float* lut = new float[length];

  size_t n_bins = _bins.size();

  // Build LUT from cumulative histrogram

  // Find first non-zero bin
  int i = 0;
  while (_bins[i] == 0)
    ++i;

  if (_bins[i] == _pixelCount) {
    for (size_t x = 0; x < length; ++x) {
      float fx = (float)x / (float)(length - 1);
      lut[x] = (float)_pixelCount / (float)n_bins; // or (n_bins-1) ??
    }
    return lut;
  }

  // Compute scale
  float scale = 1.0f / (_pixelCount - _bins[i]);

  // Initialize lut
  for (size_t x = 0; x < length; ++x) {
    float fx = (float)x / (float)(length - 1);
    // select the cumulative value from histo
    uint32_t sum = _ccounts[int(fx * (_ccounts.size() - 1))];
    // the value is saturated in range [0, max_val]
    lut[x] = std::max(0.0f, std::min(sum * scale, 1.0f));
  }
  return lut;
}

float*
Histogram::generateFromGradientData(const GradientData& gradientData, size_t length) const
{
  switch (gradientData.m_activeMode) {
    case GradientEditMode::WINDOW_LEVEL:
      return generate_windowLevel(gradientData.m_window, gradientData.m_level, length);
    case GradientEditMode::PERCENTILE:
      return generate_percentiles(gradientData.m_pctLow, gradientData.m_pctHigh, length);
    case GradientEditMode::MINMAX: {
      // min and max are already set in gradientData
      float lowEnd = normalizeInt(gradientData.m_minu16);
      float highEnd = normalizeInt(gradientData.m_maxu16);
      float window = highEnd - lowEnd;
      float level = (lowEnd + highEnd) * 0.5f;
      return generate_windowLevel(window, level, length);
    }
    case GradientEditMode::ISOVALUE: {
      float lowEnd = gradientData.m_isovalue - gradientData.m_isorange * 0.5f;
      float highEnd = gradientData.m_isovalue + gradientData.m_isorange * 0.5f;
      std::vector<LutControlPoint> pts;
      pts.push_back({ 0.0f, 0.0f });
      pts.push_back({ lowEnd, 0.0f });
      pts.push_back({ lowEnd, 1.0f });
      pts.push_back({ highEnd, 1.0f });
      pts.push_back({ highEnd, 0.0f });
      pts.push_back({ 1.0f, 0.0f });
      return generate_controlPoints(pts, length);
    }
    case GradientEditMode::CUSTOM:
      return generate_controlPoints(gradientData.m_customControlPoints, length);
    default:
      return generate_fullRange(length);
  }
}
