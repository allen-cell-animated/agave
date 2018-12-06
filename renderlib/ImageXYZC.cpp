#include "ImageXYZC.h"

#include "Logging.h"

#include <QCoreApplication>
#include <QThread>

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
                     float sz)
  : m_x(x)
  , m_y(y)
  , m_z(z)
  , m_c(c)
  , m_bpp(bpp)
  , m_data(data)
  , m_scaleX(sx)
  , m_scaleY(sy)
  , m_scaleZ(sz)
{
  for (uint32_t i = 0; i < m_c; ++i) {
    m_channels.push_back(new Channelu16(x, y, z, reinterpret_cast<uint16_t*>(ptr(i))));
  }
  for (uint32_t i = 0; i < m_c; ++i) {
    m_channels[i]->generateGradientMagnitudeVolume(physicalSizeX(), physicalSizeY(), physicalSizeZ());

    LOG_INFO << "Channel " << i << ":" << (m_channels[i]->m_min) << "," << (m_channels[i]->m_max);
    // LOG_INFO << "gradient range " << i << ":" << (_channels[i]->_gradientMagnitudeMin) << "," <<
    // (_channels[i]->_gradientMagnitudeMax);
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
ImageXYZC::setChannelNames(std::vector<QString>& channelNames)
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
  m_scaleX = x;
  m_scaleY = y;
  m_scaleZ = z;
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
ImageXYZC::getDimensions() const
{
  // Compute physical size
  const glm::vec3 PhysicalSize(
    physicalSizeX() * (float)sizeX(), physicalSizeY() * (float)sizeY(), physicalSizeZ() * (float)sizeZ());
  // glm::gtx::component_wise::compMax(PhysicalSize);
  float m = std::max(PhysicalSize.x, std::max(PhysicalSize.y, PhysicalSize.z));

  // Compute the volume's max extent - scaled to max dimension.
  return PhysicalSize / m;
}

// count is how many elements to walk for input and output.
FuseWorkerThread::FuseWorkerThread(size_t thread_idx,
                                   size_t nthreads,
                                   uint8_t* outptr,
                                   const ImageXYZC* img,
                                   const std::vector<glm::vec3>& colors)
  : m_thread_idx(thread_idx)
  , m_nthreads(nthreads)
  , m_outptr(outptr)
  , m_channelColors(colors)
  , m_img(img)
{
  // size_t num_pixels = _img->sizeX() * _img->sizeY() * _img->sizeZ();
  // num_pixels /= _nthreads;
  // assert(num_pixels * _nthreads == _img->sizeX() * _img->sizeY() * _img->sizeZ());
}
void
FuseWorkerThread::run()
{
  float value = 0;
  float r = 0, g = 0, b = 0;
  uint8_t ar = 0, ag = 0, ab = 0;

  size_t num_total_pixels = m_img->sizeX() * m_img->sizeY() * m_img->sizeZ();
  size_t num_pixels = num_total_pixels / m_nthreads;
  // last one gets the extras.
  if (m_thread_idx == m_nthreads - 1) {
    num_pixels += num_total_pixels % m_nthreads;
  }

  size_t ncolors = m_channelColors.size();
  size_t nch = std::min((size_t)m_img->sizeC(), ncolors);

  uint8_t* outptr = m_outptr;
  outptr += ((num_total_pixels / m_nthreads) * 3 * m_thread_idx);

  for (uint32_t i = 0; i < nch; ++i) {
    glm::vec3 c = m_channelColors[i];
    if (c == glm::vec3(0, 0, 0)) {
      continue;
    }
    r = c.x; // 0..1
    g = c.y;
    b = c.z;
    uint16_t* channeldata = reinterpret_cast<uint16_t*>(m_img->ptr(i));
    // jump to offset for this thread.
    channeldata += ((num_total_pixels / m_nthreads) * m_thread_idx);

    // array of 256 floats
    float* lut = m_img->channel(i)->m_lut;
    float chmax = (float)m_img->channel(i)->m_max;
    float chmin = (float)m_img->channel(i)->m_min;
    // lut = luts[idx][c.enhancement];

    for (size_t cx = 0, fx = 0; cx < num_pixels; cx++, fx += 3) {
      value = (float)(channeldata[cx] - chmin) / (float)(chmax - chmin);
      // value = (float)channeldata[cx] / 65535.0f;
      value = lut[(int)(value * 255.0 + 0.5)]; // 0..255

      // what if rgb*value > 1?
      ar = outptr[fx + 0];
      outptr[fx + 0] = std::max(ar, static_cast<uint8_t>(r * value * 255));
      ag = outptr[fx + 1];
      outptr[fx + 1] = std::max(ag, static_cast<uint8_t>(g * value * 255));
      ab = outptr[fx + 2];
      outptr[fx + 2] = std::max(ab, static_cast<uint8_t>(b * value * 255));
    }
  }

  emit resultReady(m_thread_idx);
}

// fuse: fill volume of color data, plus volume of gradients
// n channels with n colors: use "max" or "avg"
// n channels with gradients: use "max" or "avg"
void
ImageXYZC::fuse(const std::vector<glm::vec3>& colorsPerChannel,
                uint8_t** outRGBVolume,
                uint16_t** outGradientVolume) const
{
  // todo: this can easily be a cuda kernel that loops over channels and does a max operation, if it has the full volume
  // data in gpu mem.

  // create and zero
  uint8_t* rgbVolume = *outRGBVolume;
  memset(*outRGBVolume, 0, 3 * m_x * m_y * m_z * sizeof(uint8_t));

  const bool FUSE_THREADED = true;
  if (FUSE_THREADED) {

    const size_t NTHREADS = 4;
    // set a bit for each thread as they complete
    uint32_t done = 0;
    FuseWorkerThread** workers = new FuseWorkerThread*[NTHREADS];
    for (size_t i = 0; i < NTHREADS; ++i) {
      workers[i] = new FuseWorkerThread(i, NTHREADS, rgbVolume, this, colorsPerChannel);
      QObject::connect(
        workers[i], &FuseWorkerThread::resultReady, [&done](size_t whichThread) { done |= (1 << whichThread); });
      QObject::connect(workers[i], &FuseWorkerThread::finished, workers[i], &QObject::deleteLater);
      workers[i]->start();
    }
    // WAIT FOR ALL.
    // (1 << 4) - 1 = 10000 -1 = 01111
    //		while (done < ((uint32_t)1 << NTHREADS) - (uint32_t)1) {
    //		}
    for (size_t i = 0; i < NTHREADS; ++i) {
      workers[i]->wait();
    }
    assert(done == ((uint32_t)1 << NTHREADS) - (uint32_t)1);
    delete[] workers;
    // Instead of waiting, handle completion in the resultReady callback.
    // when a new fuse call comes in, and fuse threads are currently active, then queue it:
    // if there is already a fuse waiting to happen, replace it with the new req.
    // when fuse is done, check to see if there's a queued one.
  } else {

    float value = 0;
    float r = 0, g = 0, b = 0;
    uint8_t ar = 0, ag = 0, ab = 0;

    size_t ncolors = colorsPerChannel.size();
    size_t nch = std::min((size_t)m_c, ncolors);
    for (uint32_t i = 0; i < nch; ++i) {
      glm::vec3 c = colorsPerChannel[i];
      if (c == glm::vec3(0, 0, 0)) {
        continue;
      }
      r = c.x; // 0..1
      g = c.y;
      b = c.z;
      uint16_t* channeldata = reinterpret_cast<uint16_t*>(ptr(i));

      // array of 256 floats
      float* lut = this->channel(i)->m_lut;
      float chmax = (float)this->channel(i)->m_max;
      // lut = luts[idx][c.enhancement];

      for (size_t cx = 0, fx = 0; cx < m_x * m_y * m_z; cx++, fx += 3) {
        value = (float)channeldata[cx] / chmax;
        // value = (float)channeldata[cx] / 65535.0f;
        value = lut[(int)(value * 255.0 + 0.5)]; // 0..255

        // what if rgb*value > 1?
        ar = rgbVolume[fx + 0];
        rgbVolume[fx + 0] = std::max(ar, static_cast<uint8_t>(r * value * 255));
        ag = rgbVolume[fx + 1];
        rgbVolume[fx + 1] = std::max(ag, static_cast<uint8_t>(g * value * 255));
        ab = rgbVolume[fx + 2];
        rgbVolume[fx + 2] = std::max(ab, static_cast<uint8_t>(b * value * 255));
      }
    }
  }
  /*
outGradientVolume = new uint16_t[_x * _y * _z];
memset(outGradientVolume, 0, _x*_y*_z*sizeof(uint16_t));

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
*/
}

// 3d median filter?

Channelu16::Channelu16(uint32_t x, uint32_t y, uint32_t z, uint16_t* ptr)
  : m_histogram(ptr, x * y * z)
  , m_window(1.0f)
  , m_level(0.5f)
{
  m_gradientMagnitudePtr = nullptr;
  m_ptr = ptr;

  m_x = x;
  m_y = y;
  m_z = z;

  m_min = m_histogram._dataMin;
  m_max = m_histogram._dataMax;

  //_lut = _histogram.generate_auto2(_window, _level);
  m_lut = m_histogram.initialize_thresholds();
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
