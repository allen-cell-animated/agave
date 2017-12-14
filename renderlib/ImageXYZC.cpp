#include "ImageXYZC.h"

#include "Logging.h"

#include <math.h>
#include <algorithm>
#include <sstream>

template<class T>
const T& clamp(const T& v, const T& lo, const T& hi)
{
	assert(hi > lo);
	return (v<lo) ? lo : (v>hi ? hi : v);
}

ImageXYZC::ImageXYZC(uint32_t x, uint32_t y, uint32_t z, uint32_t c, uint32_t bpp, uint8_t* data, float sx, float sy, float sz)
	: _x(x), _y(y), _z(z), _c(c), _bpp(bpp), _data(data), _scaleX(sx), _scaleY(sy), _scaleZ(sz)
{
	for (uint32_t i = 0; i < _c; ++i) {
		_channels.push_back(new Channelu16(x, y, z, reinterpret_cast<uint16_t*>(ptr(i))));
	}
}

ImageXYZC::~ImageXYZC()
{
	for (uint32_t i = 0; i < _c; ++i) {
		delete _channels[i];
		_channels[i] = nullptr;
	}
	delete[] _data;
}

uint32_t ImageXYZC::sizeX() const
{
	return _x;
}

uint32_t ImageXYZC::sizeY() const
{
	return _y;
}

uint32_t ImageXYZC::sizeZ() const
{
	return _z;
}

float ImageXYZC::physicalSizeX() const
{
	return _scaleX;
}

float ImageXYZC::physicalSizeY() const
{
	return _scaleY;
}

float ImageXYZC::physicalSizeZ() const
{
	return _scaleZ;
}

uint32_t ImageXYZC::sizeC() const
{
	return _c;
}

uint32_t ImageXYZC::sizeOfElement() const
{
	return _bpp/8;
}

size_t ImageXYZC::sizeOfPlane() const
{
	return _x * _y * sizeOfElement();
}

size_t ImageXYZC::sizeOfChannel() const
{
	return sizeOfPlane() * _z;
}

size_t ImageXYZC::size() const
{
	return sizeOfChannel() * _c;
}

uint8_t* ImageXYZC::ptr(uint32_t channel, uint32_t z) const
{
	// advance ptr by this amount of uint8s.
	return _data + ((channel*sizeOfChannel()) + (z*sizeOfPlane()));
}

Channelu16* ImageXYZC::channel(uint32_t channel) const
{
	return _channels[channel];
}

// fuse: generate volume of color data, plus volume of gradients
// n channels with n colors: use "max" or "avg"
// n channels with gradients: use "max" or "avg"
void ImageXYZC::fuse(const std::vector<glm::vec3>& colorsPerChannel, uint8_t* outRGBVolume, uint16_t* outGradientVolume)
{
	//todo: this can easily be a cuda kernel that loops over channels and does a max operation, if it has the full volume data in gpu mem.

	// create and zero
	outRGBVolume = new uint8_t[3 * _x * _y * _z];
	memset(outRGBVolume, 0, 3 * _x*_y*_z * sizeof(uint8_t));
	outGradientVolume = new uint16_t[_x * _y * _z];
	memset(outGradientVolume, 0, _x*_y*_z*sizeof(uint16_t));

	uint16_t value = 0;
	float r=0, g=0, b=0;
	uint8_t ar = 0, ag = 0, ab = 0;

	for (uint32_t i = 0; i < _c; ++i) {
		glm::vec3 c = colorsPerChannel[i];
		if (c == glm::vec3(0,0,0)) {
			continue;
		}
		r = c.x; // 0..1
		g = c.y;
		b = c.z;
		uint16_t* channeldata = reinterpret_cast<uint16_t*>(ptr(i));
		//lut = luts[idx][c.enhancement];

		for (size_t cx = 0, fx = 0; cx < _x*_y*_z; cx ++, fx += 3) {
			value = channeldata[cx];  
			//value = lut[value]; // 0..255

			// what if rgb*value > 255?
			ar = outRGBVolume[fx + 0];
			outRGBVolume[fx + 0] = std::max(ar, static_cast<uint8_t>(r * value));
			ag = outRGBVolume[fx + 1];
			outRGBVolume[fx + 1] = std::max(ag, static_cast<uint8_t>(g * value));
			ab = outRGBVolume[fx + 2];
			outRGBVolume[fx + 2] = std::max(ab, static_cast<uint8_t>(b * value));
		}
	}

	// todo: gradient fusion
	for (uint32_t i = 0; i < _c; ++i) {
		glm::vec3 c = colorsPerChannel[i];
		if (c == glm::vec3(0, 0, 0)) {
			continue;
		}
		// get gradient data for channel
		uint16_t* gradientData = reinterpret_cast<uint16_t*>(ptr(i));
		//lut = luts[idx][c.enhancement];
		
		for (size_t cx = 0; cx < _x*_y*_z; cx++) {
			outGradientVolume[cx] = std::max(outGradientVolume[cx], gradientData[cx]);
		}
	}
}

// 3d median filter?

Channelu16::Channelu16(uint32_t x, uint32_t y, uint32_t z, uint16_t* ptr)
	: _histogram(ptr, x*y*z)
{
	_gradientMagnitudePtr = nullptr;
	_ptr = ptr;

	_x = x;
	_y = y;
	_z = z;

	_min = _histogram._dataMin;
	_max = _histogram._dataMax;
	_lut = _histogram.generate_auto2();
}

Channelu16::~Channelu16() 
{
	delete[] _lut;
	delete[] _gradientMagnitudePtr;
}

uint16_t* Channelu16::generateGradientMagnitudeVolume(float scalex, float scaley, float scalez) {
	float maxspacing = std::max(scalex, std::max(scaley, scalez));
	float xspacing = scalex / maxspacing;
	float yspacing = scaley / maxspacing;
	float zspacing = scalez / maxspacing;

	uint16_t* outptr = new uint16_t[_x*_y*_z];
	_gradientMagnitudePtr = outptr;

	int useZmin, useZmax, useYmin, useYmax, useXmin, useXmax;

	double d, sum;

	// deltaz is one plane of data (x*y pixels)
	const int32_t dz = _x*_y;
	// deltay is one row of data (x pixels)
	const int32_t dy = _x;
	// deltax is one pixel
	const int32_t dx = 1;

	uint16_t* inptr = _ptr;
	for (uint32_t z = 0; z < _z; ++z) {
		useZmin = (z <= 0) ? 0 : -dz;
		useZmax = (z >= _z-1) ? 0 : dz;
		for (uint32_t y = 0; y < _y; ++y) {
			useYmin = (y <= 0) ? 0 : -dy;
			useYmax = (y >= _y - 1) ? 0 : dy;
			for (uint32_t x = 0; x < _x; ++x) {
				useXmin = (x <= 0) ? 0 : -dx;
				useXmax = (x >= _x - 1) ? 0 : dx;

				d = static_cast<double>(inptr[useXmin]);
				d -= static_cast<double>(inptr[useXmax]);
				d /= xspacing; // divide or multiply here??
				sum = d*d;

				d = static_cast<double>(inptr[useYmin]);
				d -= static_cast<double>(inptr[useYmax]);
				d /= yspacing; // divide or multiply here??
				sum += d*d;

				d = static_cast<double>(inptr[useZmin]);
				d -= static_cast<double>(inptr[useZmax]);
				d /= zspacing; // divide or multiply here??
				sum += d*d;

				*outptr = static_cast<uint16_t>(sqrt(sum));
				outptr++;
				inptr++;
			}
		}
	}

	return outptr;
}


Histogram::Histogram(uint16_t* data, size_t length, size_t bins)
	: _bins(bins)
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
	float fval;
	float range = (float)(_dataMax - _dataMin);
	float bin = (float)(bins - 1);
	for (size_t i = 0; i < length; ++i) {
		val = data[i];
		// normalize to 0..1 range
		// ZERO BIN is _dataMin intensity!!!!!! _dataMin MIGHT be nonzero.
		fval = (float)(val - _dataMin) / range;
		// select a bin
		fval *= bin;
		// discretize (drop the fractional part?)
		_bins[(size_t)fval] ++;
		// bins goes from min to max of data range. not datatype range.
	}

	// total number of pixels minus the number of zero pixels
	_nonzeroPixelCount = length - _bins[0];

	// get the bin with the most frequently occurring NONZERO value
	_maxBin = 1;
	uint32_t curmax = _bins[1];
	for (size_t i = 1; i < _bins.size(); i++) {
		if (_bins[i] > curmax) {
			_maxBin = i;
			curmax = _bins[i];
		}
	}
}

float* Histogram::generate_fullRange(size_t length) {
	return generate_windowLevel(1.0, 0.5, length);
}

float* Histogram::generate_dataRange(size_t length) {
	return generate_windowLevel(1.0, 0.5, length);
}

float* Histogram::generate_bestFit(size_t length) {
	size_t pixcount = _nonzeroPixelCount;
	size_t limit = pixcount / 10;

	size_t i = 0;
	size_t count = 0;
	for (i = 1; i < _bins.size(); ++i) {
		count += _bins[i];
		if (count > limit) {
			break;
		}
	}
	size_t hmin = i;

	count = 0;
	for (i = _bins.size() - 1; i >= 1; --i) {
		count += _bins[i];
		if (count > limit) {
			break;
		}
	}
	size_t hmax = i;

	size_t range = hmax - hmin;
	if (range < 1) {
		range = 256;
	}

	return generate_windowLevel((float)(range) / (float)length,
		((float)hmin + (float)range*0.5f) / (float)length, length);
}

// attempt to redo imagej's Auto
float* Histogram::generate_auto2(size_t length) {

	size_t AUTO_THRESHOLD = 5000;
	size_t pixcount = _nonzeroPixelCount;
	//  const pixcount = this.imgData.data.length;
	size_t limit = pixcount / 10;
	size_t threshold = pixcount / AUTO_THRESHOLD;

	// this will skip the "zero" bin which contains pixels of zero intensity.
	size_t hmin = _bins.size() - 1;
	size_t hmax = 1;
	for (size_t i = 1; i < _bins.size(); ++i) {
		if (_bins[i] > threshold && _bins[i] <= limit) {
			hmin = i;
			break;
		}
	}
	for (size_t i = _bins.size() - 1; i >= 1; --i) {
		if (_bins[i] > threshold && _bins[i] <= limit) {
			hmax = i;
			break;
		}
	}

	if (hmax < hmin) {
		// just reset to whole range in this case.
		return generate_fullRange(length);
	}
	else {
		float range = (float)hmax - (float)hmin;
		return generate_windowLevel((range) / (float)length,
			((float)hmin + range*0.5f) / (float)length, length);
	}
}

float* Histogram::generate_auto(size_t length) {

	// simple linear mapping cutting elements with small appearence
	// get 10% threshold
	float PERCENTAGE = 0.1f;
	float th = std::floor(_bins[_maxBin] * PERCENTAGE);
	size_t b = 0;
	size_t e = _bins.size() - 1;
	for (size_t x = 1; x < _bins.size(); ++x) {
		if (_bins[x] > th) {
			b = x;
			break;
		}
	}
	for (size_t x = _bins.size() - 1; x >= 1; --x) {
		if (_bins[x] > th) {
			e = x;
			break;
		}
	}

	size_t range = e - b;
	if (range < 1) {
		range = 256;
	}
	// if range == e-b, then
	// b+range/2 === e-range/2 
	// 
	return generate_windowLevel((float)range / (float)length, ((float)b + (float)range*0.5f) / (float)length, length);
}

// window and level are percentages of full range 0..1
float* Histogram::generate_windowLevel(float window, float level, size_t length)
{
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
		float v = ((float)x/(float)length - a) / ((float)range);
		lut[x] = clamp(v, 0.0f, 1.0f);
	}

	return lut;

}

void Channelu16::debugprint() {
	// stringify for output
	std::stringstream ss;
	for (size_t x = 0; x < 256; ++x) {
		ss << _lut[x] << ", ";
	}
	LOG_DEBUG << "LUT: " << ss.str();
}
