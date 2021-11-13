#include "Fuse.h"

#include "ImageXYZC.h"

#include <thread>

// fuse: fill volume of color data, plus volume of gradients
// n channels with n colors: use "max" or "avg"
// n channels with gradients: use "max" or "avg"
void
Fuse::fuse(const ImageXYZC* img,
           const std::vector<glm::vec3>& colorsPerChannel,
           uint8_t** outRGBVolume,
           uint16_t** outGradientVolume)
{
  // todo: this can easily be a cuda kernel that loops over channels and does a max operation, if it has the full volume
  // data in gpu mem.

  // create and zero
  uint8_t* rgbVolume = *outRGBVolume;
  memset(*outRGBVolume, 0, 3 * img->sizeX() * img->sizeY() * img->sizeZ() * sizeof(uint8_t));

  const bool FUSE_THREADED = true;
  if (FUSE_THREADED) {

    const size_t NTHREADS = 4;
    std::vector<std::thread> workers;
    for (size_t i = 0; i < NTHREADS; ++i) {
      workers.emplace_back(std::thread([i, NTHREADS, &rgbVolume, &img, &colorsPerChannel]() {
        FuseWorkerThread t(i, NTHREADS, rgbVolume, img, colorsPerChannel);
        t.run();
      }));
    }
    // WAIT FOR ALL.
    for (auto& worker : workers) {
      worker.join();
    }

    // THIS IS TOO SLOW AS IS.
    // TODO:
    // Instead of waiting, handle completion in an atomic counter or some kind of signalling.
    // when a new fuse call comes in, and fuse threads are currently active, then queue it:
    // if there is already a fuse waiting to happen, replace it with the new req.
    // when fuse is done, check to see if there's a queued one.
  } else {

    float value = 0;
    float r = 0, g = 0, b = 0;
    uint8_t ar = 0, ag = 0, ab = 0;

    size_t ncolors = colorsPerChannel.size();
    size_t nch = std::min((size_t)img->sizeC(), ncolors);
    for (uint32_t i = 0; i < nch; ++i) {
      glm::vec3 c = colorsPerChannel[i];
      if (c == glm::vec3(0, 0, 0)) {
        continue;
      }
      r = c.x; // 0..1
      g = c.y;
      b = c.z;
      uint16_t* channeldata = reinterpret_cast<uint16_t*>(img->ptr(i));

      // array of 256 floats
      float* lut = img->channel(i)->m_lut;
      float chmax = (float)img->channel(i)->m_max;
      // lut = luts[idx][c.enhancement];

      for (size_t cx = 0, fx = 0; cx < img->sizeX() * img->sizeY() * img->sizeZ(); cx++, fx += 3) {
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

  // emit resultReady(m_thread_idx);
}
