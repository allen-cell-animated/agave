#include "Histogram.h"

#include "Logging.h"

#include <algorithm>
#include <numeric>

template<class T>
const T& clamp(const T& v, const T& lo, const T& hi)
{
	assert(hi > lo);
	return (v<lo) ? lo : (v>hi ? hi : v);
}

Histogram::Histogram(uint16_t* data, size_t length, size_t num_bins)
	: _bins(num_bins), _ccounts(num_bins)
{
	std::fill(_bins.begin(), _bins.end(), 0);

	_dataMin = data[0];
	_dataMax = data[0];

	uint16_t val;
	for (size_t i = 0; i < length; ++i) {
		val = data[i];
		if (val > _dataMax) {
			_dataMax = val;
		}
		else if (val < _dataMin) {
			_dataMin = val;
		}
	}
//	float fval;
	float range = (float)(_dataMax - _dataMin);
	float binmax = (float)(num_bins - 1);
	for (size_t i = 0; i < length; ++i) {
		size_t whichbin = (size_t) ( (float)(data[i] - _dataMin) / range * binmax + 0.5 );
//		val = data[i];
//		// normalize to 0..1 range
//		// ZERO BIN is _dataMin intensity!!!!!! _dataMin MIGHT be nonzero.
//		fval = (float)(val - _dataMin) / range;
//		// select a bin
//		fval *= binmax;
//		// discretize (drop the fractional part?)
//		size_t whichbin = (size_t)fval;
		_bins[whichbin] ++;
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

float* Histogram::generate_fullRange(float& window, float& level, size_t length) {
	window = 1.0;
	level = 0.5;
	return generate_windowLevel(window, level, length);
}

float* Histogram::generate_dataRange(float& window, float& level, size_t length) {
	window = 1.0;
	level = 0.5;
	return generate_windowLevel(window, level, length);
}

float* Histogram::generate_bestFit(float& window, float& level, size_t length) {
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
		range = _bins.size()-1;
	}

	window = (float)(range) / (float)(_bins.size()-1);
	level = ((float)hmin + (float)range*0.5f) / (float)(_bins.size() -1);
	return generate_windowLevel(window, level, length);
}

// attempt to redo imagej's Auto
float* Histogram::generate_auto2(float& window, float& level, size_t length) {

	size_t AUTO_THRESHOLD = 10000;
	size_t pixcount = _pixelCount;// -_bins[0];
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

//#if 0
	// this will skip the "zero" bin which contains pixels of zero intensity.
	size_t hmin = nbins - 1;
	size_t hmax = 0;
	for (size_t i = 0; i < nbins; ++i) {
		if (_bins[i] > threshold && _bins[i] <= limit) {
			hmin = i;
			break;
		}
	}
	for (size_t i = nbins - 1; i >= 0; --i) {
		if (_bins[i] > threshold && _bins[i] <= limit) {
			hmax = i;
			break;
		}
	}
//#endif

	if (hmax < hmin) {
		// just reset to whole range in this case.
		return generate_fullRange(window, level, length);
	}
	else {
		LOG_DEBUG << "auto2 range: " << hmin << "..." << hmax;
		float range = (float)hmax - (float)hmin;
		window = (range) / (float)(nbins-1);
		level = ((float)hmin + range*0.5f) / (float)(nbins-1);
		LOG_DEBUG << "auto2 window/level: " << window << " / " << level;
		return generate_windowLevel(window, level, length);
	}
}

float* Histogram::generate_auto(float& window, float& level, size_t length) {

	// simple linear mapping cutting elements with small appearence
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
	window = (float)range / (float)(_bins.size() -1);
	level = ((float)b + (float)range*0.5f) / (float)(_bins.size() -1);
	return generate_windowLevel(window, level, length);
}

// window and level are percentages of full range 0..1
float* Histogram::generate_windowLevel(float window, float level, size_t length)
{
	LOG_DEBUG << "window/level: " << window << ", " << level;

	// return a LUT with new values(?)
	// data type of lut values is out_phys_range (uint8)
	// length of lut is number of histogram bins (represents the input data range)
	float* lut = new float[length];

	float a = (level - window*0.5f);
	float b = (level + window*0.5f);
	// b-a should be equal to window!
	assert(fabs(b - a - window) < 0.0001);
	float range = window;
	//float range = b - a;

	for (size_t x = 0; x < length; ++x) {
		float v = ((float)x/(float)(length-1) - a) / range;
		lut[x] = clamp(v, 0.0f, 1.0f);
	}

	return lut;

}

float* Histogram::generate_controlPoints(std::vector<std::pair<float, float>> pts, size_t length) {
	// pts is piecewise linear from first to last control point.
	// pts is in order of increasing x value (the first element of the pair)
	// pts[0].first === 0
	// pts[pts.size()-1].first === 1

	float* lut = new float[length];


	for (size_t x = 0; x < length; ++x) {
		float fx = (float)x / (float)(length - 1);
		// find the interval of control points that contains fx.
		for (size_t i = 0; i < pts.size()-1; ++i) {
			// am i in between?
			if ((fx >= pts[i].first) && (fx <= pts[i + 1].first)) {
				// what fraction of this interval in x?
				float fxi = (fx - pts[i].first) / (pts[i+1].first - pts[i].first);
				// use that fraction against y range
				lut[x] = fxi * (pts[i + 1].second - pts[i].second);
				break;
			}
		}
	}
	return lut;
}

void Histogram::bin_range(uint32_t nbins, float& firstBinCenter, float& lastBinCenter, float& binSize)
{
	uint16_t dmin = _dataMin;
	uint16_t dmax = _dataMax;
	float fbc, lbc, bsize;
	if (nbins > 1) {
		if (dmax > dmin) {
			fbc = dmin;
			lbc = dmax;
		}
		else {
			fbc = dmin - 1.0f;
			lbc = dmax + 1.0f;
		}
		bsize = (lbc - fbc) / (nbins - 1);
	}
	else {
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
std::vector<uint32_t> Histogram::bin_counts(uint32_t nbins)
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
		float fb0 = s + (b - 0.5f)*r;
		int b0 = int(ceil(fb0));
		float f0 = b0 - fb0;
		if (b0 < 0) {
			b0 = 0;
			f0 = 0;
		}
		float fb1 = s + (b + 0.5f)*r;
		int b1 = int(floor(fb1));
		float f1 = fb1 - b1;
		if (b1 >= (int32_t)fbins) {
			b1 = fbins;
			f1 = 0;
		}
		uint32_t c = 0;
		if ((b0 - 1) == b1) {
			c += (uint32_t)(r*fcounts[b0 - 1]);
		}
		else {
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
float Histogram::rank_data_value(float fraction)
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

float* Histogram::initialize_thresholds(float vfrac_min /*= 0.01*/, float vfrac_max /*= 0.90*/)
{
	float ilow = 0.0f;
	float imid = 0.8f;
	float imax = 1.0f;
	float vlow = rank_data_value(1.0f - vfrac_max);
	float vmid = rank_data_value(1.0f - vfrac_min);
	float vmax = _dataMax;
	LOG_DEBUG << "LOW: " << vlow << " HIGH: " << vmid;

	// normalize to 0..1
	vlow = (vlow - _dataMin) / (_dataMax - _dataMin);
	vmid = (vmid - _dataMin) / (_dataMax - _dataMin);
	vmax = 1.0;

	if ((vlow < vmid) && (vmid < vmax)) {
		return generate_controlPoints({ {0.0f,0.0f}, {vlow, ilow}, {vmid, imid}, {vmax, imax} });
	}
	else {
		return generate_controlPoints({ {0.0f,0.0f}, { vlow, ilow }, { 0.9f*vlow + 0.1f*vmax, imid }, { vmax, imax } });
	}

}

float* Histogram::generate_equalized(size_t length)
{
	float* lut = new float[length];

	size_t n_bins = _bins.size();

	// Build LUT from cumulative histrogram

	// Find first non-zero bin
	int i = 0;
	while (_bins[i] == 0) ++i;

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
		uint32_t sum = _ccounts[int(fx*(_ccounts.size() - 1))];
		// the value is saturated in range [0, max_val]
		lut[x] = std::max(0.0f, std::min(sum * scale, 1.0f));
	}
	return lut;
}
