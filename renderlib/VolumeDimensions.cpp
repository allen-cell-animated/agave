#include "VolumeDimensions.h"

#include "Logging.h"

#include <set>

bool
startsWith(std::string mainStr, std::string toMatch)
{
  // std::string::find returns 0 if toMatch is found at starting
  if (mainStr.find(toMatch) == 0)
    return true;
  else
    return false;
}

bool
VolumeDimensions::validate() const
{
  bool ok = true;
  if (dimensionOrder == "") {
    LOG_ERROR << "Dimension order is null";
    ok = false;
  }
  if (!startsWith(dimensionOrder, "XY") && !startsWith(dimensionOrder, "YX")) {
    LOG_ERROR << "Invalid dimension order: " << dimensionOrder;
    ok = false;
  }
  // legal characters in DimensionOrder:
  static const std::set<char> standardDims = { 'X', 'Y', 'Z', 'C', 'T' };
  // check for any dups or extra
  std::set<char> dimsInDimOrder;
  bool dups = false;
  bool badDims = false;
  for (auto d : dimensionOrder) {
    if (dimsInDimOrder.count(d) == 0) {
      dimsInDimOrder.insert(d);
    } else {
      dups = true;
    }
    if (standardDims.count(d) == 0) {
      badDims = true;
    }
  }
  if (dups || badDims) {
    LOG_ERROR << "Invalid dimension order: " << dimensionOrder;
    ok = false;
  }

  if (sizeX <= 0) {
    LOG_ERROR << "Invalid X size: " << sizeX;
    ok = false;
  }
  if (sizeY <= 0) {
    LOG_ERROR << "Invalid Y size: " << sizeY;
    ok = false;
  }
  if (sizeZ <= 0) {
    LOG_ERROR << "Invalid Z size: " << sizeZ;
    ok = false;
  }
  if (sizeC <= 0) {
    LOG_ERROR << "Invalid C size: " << sizeC;
    ok = false;
  }
  if (sizeT <= 0) {
    LOG_ERROR << "Invalid T size: " << sizeT;
    ok = false;
  }

  if (physicalSizeX <= 0) {
    LOG_ERROR << "Invalid physical pixel size x: " << physicalSizeX;
    ok = false;
  }
  if (physicalSizeY <= 0) {
    LOG_ERROR << "Invalid physical pixel size y: " << physicalSizeY;
    ok = false;
  }
  if (physicalSizeZ <= 0) {
    LOG_ERROR << "Invalid physical pixel size z: " << physicalSizeZ;
    ok = false;
  }

  if (!channelNames.empty() && channelNames.size() != sizeC) {
    LOG_ERROR << "Invalid number of channel names: " << channelNames.size() << " for " << sizeC << " channels";
    ok = false;
  }

  if (bitsPerPixel % 8 != 0) {
    LOG_ERROR << "Bits per pixel " << bitsPerPixel << " is expected to be divisible by 8";
  }

  return ok;
}

uint32_t
VolumeDimensions::getPlaneIndex(uint32_t z, uint32_t c, uint32_t t) const
{
  size_t iz = dimensionOrder.find('Z') - 2;
  size_t ic = dimensionOrder.find('C') - 2;
  size_t it = dimensionOrder.find('T') - 2;
  // assuming dims.validate() == true :
  // assert (iz < 0 || iz > 2 || ic < 0 || ic > 2 || it < 0 || it > 2);

  // check SizeZ
  if (z < 0 || z >= sizeZ) {
    LOG_ERROR << "Invalid Z index: " << z << "/" << sizeZ;
  }

  // check SizeC
  if (c < 0 || c >= sizeC) {
    LOG_ERROR << "Invalid C index: " << c << "/" << sizeC;
  }

  // check SizeT
  if (t < 0 || t >= sizeT) {
    LOG_ERROR << "Invalid T index: " << t << "/" << sizeT;
  }

  // assign rasterization order
  int v0 = iz == 0 ? z : (ic == 0 ? c : t);
  int v1 = iz == 1 ? z : (ic == 1 ? c : t);
  int v2 = iz == 2 ? z : (ic == 2 ? c : t);
  int len0 = iz == 0 ? sizeZ : (ic == 0 ? sizeC : sizeT);
  int len1 = iz == 1 ? sizeZ : (ic == 1 ? sizeC : sizeT);

  return v0 + v1 * len0 + v2 * len0 * len1;
}

std::vector<uint32_t>
VolumeDimensions::getPlaneZCT(uint32_t planeIndex) const
{
  size_t iz = dimensionOrder.find('Z') - 2;
  size_t ic = dimensionOrder.find('C') - 2;
  size_t it = dimensionOrder.find('T') - 2;
  // assuming dims.validate() == true :
  // assert (iz < 0 || iz > 2 || ic < 0 || ic > 2 || it < 0 || it > 2);

  // check image count
  if (planeIndex < 0 || planeIndex >= sizeZ * sizeC * sizeT) {
    LOG_ERROR << "Invalid image index: " << planeIndex << "/" << (sizeZ * sizeC * sizeT);
  }

  // assign rasterization order
  int len0 = iz == 0 ? sizeZ : (ic == 0 ? sizeC : sizeT);
  int len1 = iz == 1 ? sizeZ : (ic == 1 ? sizeC : sizeT);
  // int len2 = iz == 2 ? sizeZ : (ic == 2 ? sizeC : sizeT);
  int v0 = planeIndex % len0;
  int v1 = planeIndex / len0 % len1;
  int v2 = planeIndex / len0 / len1;
  uint32_t z = iz == 0 ? v0 : (iz == 1 ? v1 : v2);
  uint32_t c = ic == 0 ? v0 : (ic == 1 ? v1 : v2);
  uint32_t t = it == 0 ? v0 : (it == 1 ? v1 : v2);

  return { z, c, t };
}

void
VolumeDimensions::log() const
{
  LOG_INFO << "Begin VolumeDimensions";
  LOG_INFO << "sizeX: " << sizeX;
  LOG_INFO << "sizeY: " << sizeY;
  LOG_INFO << "sizeZ: " << sizeZ;
  LOG_INFO << "sizeC: " << sizeC;
  LOG_INFO << "sizeT: " << sizeT;
  LOG_INFO << "DimensionOrder: " << dimensionOrder;
  LOG_INFO << "PhysicalPixelSize: [" << physicalSizeX << ", " << physicalSizeY << ", " << physicalSizeZ << "]";
  LOG_INFO << "bitsPerPixel: " << bitsPerPixel;
  LOG_INFO << "sampleFormat: " << sampleFormat;
  LOG_INFO << "End VolumeDimensions";
}
