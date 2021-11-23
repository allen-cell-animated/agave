#pragma once

#include "glm.h"

#include <atomic>
#include <condition_variable>
#include <thread>
#include <type_traits>
#include <vector>

class ImageXYZC;

// Runs a processing step that applies a color into each channel,
// and then combines the channels to result in a single RGB colored volume.
// This class should be instantiated only once per new ImageXYZC.
class Fuse
{
public:
  Fuse();
  ~Fuse();
  void init(const ImageXYZC* img, uint8_t* outRGBVolume);
  // if channel color is 0, then channel will not contribute.
  // requests a fuse operation but does not block unless Fuse class is set to single thread.
  void fuse(const std::vector<glm::vec3>& colorsPerChannel);

private:
  static const bool FUSE_THREADED;

  std::atomic_uint8_t m_nThreadsWorking = 0;
  std::vector<std::thread> m_threads;

  std::condition_variable m_conditionVar;
  std::mutex m_mutex;
  bool m_stop = false;

  // the read-only volume data
  const ImageXYZC* m_img;
  // the fused result (always rewrite to same mem location)
  uint8_t* m_outRGBVolume;

  void fuseThreadWorker(size_t whichThread, size_t nThreads);
  void doFuse(size_t whichThread, size_t nThreads, const std::vector<glm::vec3>& colors);
};

class FuseWorkerThread
{
public:
  // count is how many elements to walk for input and output.
  FuseWorkerThread(size_t thread_idx,
                   size_t nthreads,
                   uint8_t* outptr,
                   const ImageXYZC* img,
                   const std::vector<glm::vec3>& colors);
  void run();

private:
  size_t m_thread_idx;
  size_t m_nthreads;
  uint8_t* m_outptr;

  // read only!
  const ImageXYZC* m_img;
  const std::vector<glm::vec3>& m_channelColors;

  // void resultReady(size_t threadidx);
};
