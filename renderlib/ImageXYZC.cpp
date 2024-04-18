#include "ImageXYZC.h"

#include "Colormap.h"
#include "Logging.h"

#include <algorithm>
#include <math.h>
#include <sstream>

ImageXYZC::ImageXYZC(uint32_t x,
                     uint32_t y,
                     uint32_t z,
                     uint32_t c,
                     uint32_t bpp,
                     uint8_t* data,
                     float sx,
                     float sy,
                     float sz,
                     std::string spatialUnits)
  : m_x(x)
  , m_y(y)
  , m_z(z)
  , m_c(c)
  , m_bpp(bpp)
  , m_data(data)
  , m_scaleX(sx)
  , m_scaleY(sy)
  , m_scaleZ(sz)
  , m_spatialUnits(spatialUnits)
  , m_flipped(1.0, 1.0, 1.0)
{
  for (uint32_t i = 0; i < m_c; ++i) {
    m_channels.push_back(new Channelu16(x, y, z, reinterpret_cast<uint16_t*>(ptr(i))));
  }
  for (uint32_t i = 0; i < m_c; ++i) {
    LOG_INFO << "Channel " << i << ":" << (m_channels[i]->m_min) << "," << (m_channels[i]->m_max);
  }
}

ImageXYZC::~ImageXYZC()
{
  for (uint32_t i = 0; i < m_c; ++i) {
    delete m_channels[i];
    m_channels[i] = nullptr;
  }
  delete[] m_data;
}

void
ImageXYZC::setChannelNames(std::vector<std::string>& channelNames)
{
  for (uint32_t i = 0; i < m_c; ++i) {
    m_channels[i]->m_name = channelNames[i];
  }
}

uint32_t
ImageXYZC::sizeX() const
{
  return m_x;
}

uint32_t
ImageXYZC::sizeY() const
{
  return m_y;
}

uint32_t
ImageXYZC::sizeZ() const
{
  return m_z;
}

uint32_t
ImageXYZC::maxPixelDimension() const
{
  return std::max(m_x, std::max(m_y, m_z));
}

void
ImageXYZC::setPhysicalSize(float x, float y, float z)
{
  m_scaleX = abs(x);
  m_scaleY = abs(y);
  m_scaleZ = abs(z);
}

void
ImageXYZC::setVolumeAxesFlipped(float x, float y, float z)
{
  m_flipped = glm::vec3(x < 0 ? -1 : 1, y < 0 ? -1 : 1, z < 0 ? -1 : 1);
}

glm::vec3
ImageXYZC::getVolumeAxesFlipped() const
{
  return m_flipped;
}

float
ImageXYZC::physicalSizeX() const
{
  return m_scaleX;
}

float
ImageXYZC::physicalSizeY() const
{
  return m_scaleY;
}

float
ImageXYZC::physicalSizeZ() const
{
  return m_scaleZ;
}

uint32_t
ImageXYZC::sizeC() const
{
  return m_c;
}

uint32_t
ImageXYZC::sizeOfElement() const
{
  return m_bpp / 8;
}

size_t
ImageXYZC::sizeOfPlane() const
{
  return m_x * m_y * sizeOfElement();
}

size_t
ImageXYZC::sizeOfChannel() const
{
  return sizeOfPlane() * m_z;
}

size_t
ImageXYZC::size() const
{
  return sizeOfChannel() * m_c;
}

uint8_t*
ImageXYZC::ptr(uint32_t channel, uint32_t z) const
{
  // advance ptr by this amount of uint8s.
  return m_data + ((channel * sizeOfChannel()) + (z * sizeOfPlane()));
}

Channelu16*
ImageXYZC::channel(uint32_t channel) const
{
  return m_channels[channel];
}

glm::vec3
ImageXYZC::getPhysicalDimensions() const
{
  return glm::vec3(
    physicalSizeX() * (float)sizeX(), physicalSizeY() * (float)sizeY(), physicalSizeZ() * (float)sizeZ());
}

glm::vec3
ImageXYZC::getNormalizedDimensions() const
{
  // Compute physical size
  const glm::vec3 PhysicalSize = getPhysicalDimensions();
  // glm::gtx::component_wise::compMax(PhysicalSize);
  float m = std::max(PhysicalSize.x, std::max(PhysicalSize.y, PhysicalSize.z));

  // Compute the volume's max extent - scaled to max dimension.
  return PhysicalSize / m;
}

std::string
ImageXYZC::spatialUnits() const
{
  return m_spatialUnits;
}

// 3d median filter?

Channelu16::Channelu16(uint32_t x, uint32_t y, uint32_t z, uint16_t* ptr)
  : m_histogram(ptr, x * y * z)
{
  m_gradientMagnitudePtr = nullptr;
  m_ptr = ptr;

  m_x = x;
  m_y = y;
  m_z = z;

  m_min = m_histogram._dataMin;
  m_max = m_histogram._dataMax;

  m_lut = m_histogram.generate_percentiles();

  // create a hardcoded colormap to test
  m_colormap = colormapFromControlPoints(
    { ColorControlPoint(0.0f, 255u, 255u, 255u, 255u), ColorControlPoint(1.0f, 255u, 255u, 255u, 255u) });
}

void
Channelu16::updateColormap(std::vector<ColorControlPoint> stops)
{
  delete[] m_colormap;
  m_colormap = colormapFromControlPoints(stops);
}
void
Channelu16::colorize()
{
  delete[] m_colormap;
  m_colormap = colormapRandomized();
}

Channelu16::~Channelu16()
{
  delete[] m_lut;
  delete[] m_gradientMagnitudePtr;
}

uint16_t*
Channelu16::generateGradientMagnitudeVolume(float scalex, float scaley, float scalez)
{
  float maxspacing = std::max(scalex, std::max(scaley, scalez));
  float xspacing = scalex / maxspacing;
  float yspacing = scaley / maxspacing;
  float zspacing = scalez / maxspacing;

  uint16_t* outptr = new uint16_t[m_x * m_y * m_z];
  m_gradientMagnitudePtr = outptr;

  int useZmin, useZmax, useYmin, useYmax, useXmin, useXmax;

  double d, sum;

  // deltaz is one plane of data (x*y pixels)
  const int32_t dz = m_x * m_y;
  // deltay is one row of data (x pixels)
  const int32_t dy = m_x;
  // deltax is one pixel
  const int32_t dx = 1;

  uint16_t* inptr = m_ptr;
  for (uint32_t z = 0; z < m_z; ++z) {
    useZmin = (z <= 0) ? 0 : -dz;
    useZmax = (z >= m_z - 1) ? 0 : dz;
    for (uint32_t y = 0; y < m_y; ++y) {
      useYmin = (y <= 0) ? 0 : -dy;
      useYmax = (y >= m_y - 1) ? 0 : dy;
      for (uint32_t x = 0; x < m_x; ++x) {
        useXmin = (x <= 0) ? 0 : -dx;
        useXmax = (x >= m_x - 1) ? 0 : dx;

        d = static_cast<double>(inptr[useXmin]);
        d -= static_cast<double>(inptr[useXmax]);
        d /= xspacing; // divide or multiply here??
        sum = d * d;

        d = static_cast<double>(inptr[useYmin]);
        d -= static_cast<double>(inptr[useYmax]);
        d /= yspacing; // divide or multiply here??
        sum += d * d;

        d = static_cast<double>(inptr[useZmin]);
        d -= static_cast<double>(inptr[useZmax]);
        d /= zspacing; // divide or multiply here??
        sum += d * d;

        *outptr = static_cast<uint16_t>(sqrt(sum));
        outptr++;
        inptr++;
      }
    }
  }

  return outptr;
}

void
Channelu16::debugprint()
{
  // stringify for output
  std::stringstream ss;
  for (size_t x = 0; x < 256; ++x) {
    ss << m_lut[x] << ", ";
  }
  LOG_DEBUG << "LUT: " << ss.str();
}
