#pragma once

#include "Histogram.h"

#include "glm.h"

#include <QThread>

#include <inttypes.h>
#include <vector>

struct Channelu16
{
  Channelu16(uint32_t x, uint32_t y, uint32_t z, uint16_t* ptr);
  ~Channelu16();

  uint32_t m_x, m_y, m_z;

  uint16_t* m_ptr;
  uint16_t m_min;
  uint16_t m_max;

  uint16_t* m_gradientMagnitudePtr;

  Histogram m_histogram;
  float* m_lut;

  uint16_t* generateGradientMagnitudeVolume(float scalex, float scaley, float scalez);

  void generateFromGradientData(const GradientData& gradientData)
  {
    delete[] m_lut;
    m_lut = m_histogram.generateFromGradientData(gradientData);
  }

  void generate_windowLevel(float window, float level)
  {
    delete[] m_lut;
    m_lut = m_histogram.generate_windowLevel(window, level);
  }
  void generate_auto2()
  {
    delete[] m_lut;
    m_lut = m_histogram.generate_auto2();
  }
  void generate_auto()
  {
    delete[] m_lut;
    m_lut = m_histogram.generate_auto();
  }
  void generate_bestFit()
  {
    delete[] m_lut;
    m_lut = m_histogram.generate_bestFit();
  }
  void generate_chimerax()
  {
    delete[] m_lut;
    m_lut = m_histogram.initialize_thresholds();
  }
  void generate_controlPoints(std::vector<LutControlPoint> pts)
  {
    delete[] m_lut;
    m_lut = m_histogram.generate_controlPoints(pts);
  }
  void generate_equalized()
  {
    delete[] m_lut;
    m_lut = m_histogram.generate_equalized();
  }
  void generate_percentiles(float lo = Histogram::DEFAULT_PCT_LOW, float hi = Histogram::DEFAULT_PCT_HIGH)
  {
    delete[] m_lut;
    m_lut = m_histogram.generate_percentiles(lo, hi);
  }

  void debugprint();

  QString m_name;
};

class ImageXYZC
{
public:
  ImageXYZC(uint32_t x,
            uint32_t y,
            uint32_t z,
            uint32_t c,
            uint32_t bpp,
            uint8_t* data = nullptr,
            float sx = 1.0,
            float sy = 1.0,
            float sz = 1.0);
  virtual ~ImageXYZC();

  void setPhysicalSize(float x, float y, float z);

  uint32_t sizeX() const;
  uint32_t sizeY() const;
  uint32_t sizeZ() const;
  uint32_t maxPixelDimension() const;
  float physicalSizeX() const;
  float physicalSizeY() const;
  float physicalSizeZ() const;

  glm::vec3 getDimensions() const;

  uint32_t sizeC() const;

  uint32_t sizeOfElement() const;
  size_t sizeOfPlane() const;
  size_t sizeOfChannel() const;
  size_t size() const;

  uint8_t* ptr(uint32_t channel = 0, uint32_t z = 0) const;
  Channelu16* channel(uint32_t channel) const;

  // if channel color is 0, then channel will not contribute.
  // allocates memory for outRGBVolume and outGradientVolume
  void fuse(const std::vector<glm::vec3>& colorsPerChannel, uint8_t** outRGBVolume, uint16_t** outGradientVolume) const;

  void setChannelNames(std::vector<QString>& channelNames);

private:
  uint32_t m_x, m_y, m_z, m_c, m_bpp;
  uint8_t* m_data;
  float m_scaleX, m_scaleY, m_scaleZ;
  std::vector<Channelu16*> m_channels;
};

class FuseWorkerThread : public QThread
{
  Q_OBJECT
public:
  // count is how many elements to walk for input and output.
  FuseWorkerThread(size_t thread_idx,
                   size_t nthreads,
                   uint8_t* outptr,
                   const ImageXYZC* img,
                   const std::vector<glm::vec3>& colors);
  void run() override;

private:
  size_t m_thread_idx;
  size_t m_nthreads;
  uint8_t* m_outptr;

  // read only!
  const ImageXYZC* m_img;
  const std::vector<glm::vec3>& m_channelColors;
signals:
  void resultReady(size_t threadidx);
};
