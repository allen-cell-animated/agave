#pragma once

#include "glm.h"

#include <QThread>

#include <vector>

class ImageXYZC;

// Runs a processing step that applies a color into each channel,
// and then combines the channels to result in a single RGB colored volume
class Fuse
{
public:
  // if channel color is 0, then channel will not contribute.
  // allocates memory for outRGBVolume and outGradientVolume
  static void fuse(const ImageXYZC* img,
                   const std::vector<glm::vec3>& colorsPerChannel,
                   uint8_t** outRGBVolume,
                   uint16_t** outGradientVolume);
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
