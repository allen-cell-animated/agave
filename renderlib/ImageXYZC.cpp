#include "ImageXYZC.h"

#include <math.h>
#include <algorithm>

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
{
	_gradientMagnitudePtr = nullptr;
	_ptr = ptr;

	_x = x;
	_y = y;
	_z = z;

	getMinMax(ptr, _min, _max);
}

Channelu16::~Channelu16() 
{
	delete[] _gradientMagnitudePtr;
}

void Channelu16::getMinMax(uint16_t* ptr, uint16_t& minval, uint16_t& maxval)
{
	uint16_t tmin = UINT16_MAX;
	uint16_t tmax = 0;

	uint16_t val;
	for (size_t i = 0; i < _x*_y*_z; ++i) {
		val = ptr[i];
		if (val > tmax) {
			tmax = val;
		}
		if (val < tmin) {
			tmin = val;
		}
	}
	minval = tmin;
	maxval = tmax;
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

	getMinMax(_gradientMagnitudePtr, _gradientMagnitudeMin, _gradientMagnitudeMax);
	return outptr;
}

