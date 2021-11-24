#include "Fuse.h"

#include "ImageXYZC.h"

#include <algorithm>
//#include <future>
#include <iostream>
//#include <stdexcept>
//#include <type_traits>

const bool Fuse::FUSE_THREADED = true;
const size_t Fuse::NTHREADS = 1;

Fuse::Fuse()
  : m_img(nullptr)
  , m_outRGBVolume(nullptr)
  , m_stop(false)
  , m_fuseRequests(1)
{}

Fuse::~Fuse()
{
  m_fuseRequests.unblock();
  for (auto& thread : m_threads)
    thread.join();
}

void
Fuse::init(const ImageXYZC* img, uint8_t* outRGBVolume)
{
  m_img = img;
  m_outRGBVolume = outRGBVolume;
  if (FUSE_THREADED) {
    // start the threadpool.

    for (size_t i = 0; i < NTHREADS; ++i) {
      m_threads.emplace_back(std::thread(&Fuse::fuseThreadWorker, this, i, NTHREADS));
    }
  }
}

void
Fuse::fuseThreadWorker(size_t whichThread, size_t nThreads)
{
  static const int K = 2;
  while (true) {
    std::vector<glm::vec3> data;
    for (auto n = 0; n < nThreads * K; ++n)
      if (m_fuseRequests.try_pop(data))
        break;
    if (data.empty() && !m_fuseRequests.pop(data))
      break;

    std::cout << "THREAD " << whichThread << " GOT DATA, RUNNING" << std::endl;

    doFuse(whichThread, nThreads, data);
  }

  std::cout << "THREAD " << whichThread << " LAUNCHED" << std::endl;
  // // lock to check value of m_stop
  // std::unique_lock<std::mutex> lock(m_mutex);
  // while (!m_stop) {
  //   std::cout << "THREAD " << whichThread << " IN LOOP, WAITING" << std::endl;
  //   // wait but relinquish the lock back to main thread
  //   m_conditionVar.wait(lock, [this] { return !m_queuedColorsToFuse.empty(); });

  //   std::cout << "THREAD " << whichThread << " SIGNALLED, RUNNING" << std::endl;
  //   m_nThreadsWorking++;
  //   doFuse(whichThread, nThreads, m_queuedColorsToFuse);
  //   m_nThreadsWorking--;
  //   if (m_nThreadsWorking == 0) {
  //     // i am the last thread to complete: so signal back to main thread that we have something ready?
  //     // then the main thread will re-initiate a fuse if there is a waiting request.
  //   }
  // }

  // while (!m_stop.load()) {

  //   std::unique_lock<std::mutex> lock(m_mutex);
  //   std::cout << "THREAD " << whichThread << " IN LOOP, WAITING" << std::endl;
  //   m_conditionVar.wait(lock, [this] { return !m_fuseRequests.empty() || m_stop.load(); });
  //   std::vector<glm::vec3> data = m_fuseRequests.front();
  //   m_fuseRequests.pop();
  //   lock.unlock();

  //   std::cout << "THREAD " << whichThread << " SIGNALLED, RUNNING" << std::endl;
  //   doFuse(whichThread, nThreads, data);
  // }
  std::cout << "THREAD " << whichThread << " EXITING" << std::endl;
}

// fuse: fill volume of color data, plus volume of gradients
// n channels with n colors: use "max" or "avg"
// n channels with gradients: use "max" or "avg"
void
Fuse::fuse(const std::vector<glm::vec3>& colorsPerChannel)
{
  std::cout << "fuse requested" << std::endl;
  // if threads are working, stash this request.
  // notify threads with colorsPerChannel info
  if (FUSE_THREADED) {
    std::lock_guard<std::mutex> lock(m_mutex);
    // for interactivity, we only want at most one item on the queue(?)
    if (m_fuseRequests.full()) {
      // replace existing...
      std::vector<glm::vec3> old;
      m_fuseRequests.pop(old);
      m_fuseRequests.push(colorsPerChannel);
    } else if (m_fuseRequests.empty()) {
      m_fuseRequests.push(colorsPerChannel);
    } else {
      // some kind of error, we don't want to get here.
    }
    // signal the threads w/new data
    m_conditionVar.notify_all();
  } else {
    // just run as single thread
    doFuse(0, 1, colorsPerChannel);
  }
}

// count is how many elements to walk for input and output.
void
Fuse::doFuse(size_t thread_idx, size_t nThreads, const std::vector<glm::vec3>& colors)
{
  float value = 0;
  float r = 0, g = 0, b = 0;
  uint8_t ar = 0, ag = 0, ab = 0;

  size_t num_total_pixels = m_img->sizeX() * m_img->sizeY() * m_img->sizeZ();
  size_t num_pixels = num_total_pixels / nThreads;
  // last one gets the extras.
  if (thread_idx == nThreads - 1) {
    num_pixels += num_total_pixels % nThreads;
  }

  size_t ncolors = colors.size();
  size_t nch = std::min((size_t)m_img->sizeC(), ncolors);

  uint8_t* outptr = m_outRGBVolume;
  outptr += ((num_total_pixels / nThreads) * 3 * thread_idx);

  // init with zeros first. is this needed?
  memset(outptr, 0, 3 * num_pixels * sizeof(uint8_t));

  for (uint32_t i = 0; i < nch; ++i) {
    glm::vec3 c = colors[i];
    if (c == glm::vec3(0, 0, 0)) {
      continue;
    }
    r = c.x; // 0..1
    g = c.y;
    b = c.z;
    uint16_t* channeldata = reinterpret_cast<uint16_t*>(m_img->ptr(i));
    // jump to offset for this thread.
    channeldata += ((num_total_pixels / nThreads) * thread_idx);

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
