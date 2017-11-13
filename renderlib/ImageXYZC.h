#pragma once

#include "glm.h"

#include <inttypes.h>
#include <vector>

struct Channelu16
{
	Channelu16(uint32_t x, uint32_t y, uint32_t z, uint16_t* ptr);
	~Channelu16();

	uint32_t _x, _y, _z;

	uint16_t* _ptr;
	uint16_t _min;
	uint16_t _max;

	uint16_t* _gradientMagnitudePtr;
	uint16_t _gradientMagnitudeMin;
	uint16_t _gradientMagnitudeMax;

	uint16_t* generateGradientMagnitudeVolume();
	void getMinMax(uint16_t* ptr, uint16_t& minval, uint16_t& maxval);

};

class ImageXYZC
{
public:
	ImageXYZC(uint32_t x, uint32_t y, uint32_t z, uint32_t c, uint32_t bpp, uint8_t* data=nullptr, float sx=1.0, float sy=1.0, float sz=1.0);
	virtual ~ImageXYZC();

	uint32_t sizeX() const;
	uint32_t sizeY() const;
	uint32_t sizeZ() const;
	float physicalSizeX() const;
	float physicalSizeY() const;
	float physicalSizeZ() const;

	uint32_t sizeC() const;

	uint32_t sizeOfElement() const;
	size_t sizeOfPlane() const;
	size_t sizeOfChannel() const;
	size_t size() const;

	uint8_t* ptr(uint32_t channel=0, uint32_t z=0) const;
	Channelu16* channel(uint32_t channel) const;

	// if channel color is 0, then channel will not contribute.
	// allocates memory for outRGBVolume and outGradientVolume
	void fuse(const std::vector<glm::vec3>& colorsPerChannel, uint8_t* outRGBVolume, uint16_t* outGradientVolume);

private:
	uint32_t _x, _y, _z, _c, _bpp;
	uint8_t* _data;
	float _scaleX, _scaleY, _scaleZ;
	std::vector<Channelu16*> _channels;
};

