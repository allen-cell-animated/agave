#include "Fuse.h"

#include "ImageXYZC.h"

#include <algorithm>
#include <atomic>
#include <future>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>

template<typename T>
class unbounded_queue
{
public:
  explicit unbounded_queue(bool block = true)
    : m_block{ block }
  {}

  void push(const T& item)
  {
    {
      std::scoped_lock guard(m_queue_lock);
      m_queue.push(item);
    }
    m_condition.notify_one();
  }

  void push(T&& item)
  {
    {
      std::scoped_lock guard(m_queue_lock);
      m_queue.push(std::move(item));
    }
    m_condition.notify_one();
  }

  template<typename... Args>
  void emplace(Args&&... args)
  {
    {
      std::scoped_lock guard(m_queue_lock);
      m_queue.emplace(std::forward<Args>(args)...);
    }
    m_condition.notify_one();
  }

  bool try_push(const T& item)
  {
    {
      std::unique_lock lock(m_queue_lock, std::try_to_lock);
      if (!lock)
        return false;
      m_queue.push(item);
    }
    m_condition.notify_one();
    return true;
  }

  bool try_push(T&& item)
  {
    {
      std::unique_lock lock(m_queue_lock, std::try_to_lock);
      if (!lock)
        return false;
      m_queue.push(std::move(item));
    }
    m_condition.notify_one();
    return true;
  }

  bool pop(T& item)
  {
    std::unique_lock guard(m_queue_lock);
    m_condition.wait(guard, [&]() { return !m_queue.empty() || !m_block; });
    if (m_queue.empty())
      return false;
    item = std::move(m_queue.front());
    m_queue.pop();
    return true;
  }

  bool try_pop(T& item)
  {
    std::unique_lock lock(m_queue_lock, std::try_to_lock);
    if (!lock || m_queue.empty())
      return false;
    item = std::move(m_queue.front());
    m_queue.pop();
    return true;
  }

  std::size_t size() const
  {
    std::scoped_lock guard(m_queue_lock);
    return m_queue.size();
  }

  bool empty() const
  {
    std::scoped_lock guard(m_queue_lock);
    return m_queue.empty();
  }

  void block()
  {
    std::scoped_lock guard(m_queue_lock);
    m_block = true;
  }

  void unblock()
  {
    {
      std::scoped_lock guard(m_queue_lock);
      m_block = false;
    }
    m_condition.notify_all();
  }

  bool blocking() const
  {
    std::scoped_lock guard(m_queue_lock);
    return m_block;
  }

private:
  using queue_t = std::queue<T>;
  queue_t m_queue;

  bool m_block;

  mutable std::mutex m_queue_lock;
  std::condition_variable m_condition;
};

template<typename T>
class bounded_queue
{
public:
  explicit bounded_queue(std::size_t max_size, bool block = true)
    : m_block{ block }
    , m_max_size{ max_size }
  {
    if (!m_max_size)
      throw std::invalid_argument("bad queue max-size! must be non-zero!");
  }

  bool push(const T& item)
  {
    {
      std::unique_lock guard(m_queue_lock);
      m_condition_push.wait(guard, [&]() { return m_queue.size() < m_max_size || !m_block; });
      if (m_queue.size() == m_max_size)
        return false;
      m_queue.push(item);
    }
    m_condition_pop.notify_one();
    return true;
  }

  bool push(T&& item)
  {
    {
      std::unique_lock guard(m_queue_lock);
      m_condition_push.wait(guard, [&]() { return m_queue.size() < m_max_size || !m_block; });
      if (m_queue.size() == m_max_size)
        return false;
      m_queue.push(std::move(item));
    }
    m_condition_pop.notify_one();
    return true;
  }

  template<typename... Args>
  bool emplace(Args&&... args)
  {
    {
      std::unique_lock guard(m_queue_lock);
      m_condition_push.wait(guard, [&]() { return m_queue.size() < m_max_size || !m_block; });
      if (m_queue.size() == m_max_size)
        return false;
      m_queue.emplace(std::forward<Args>(args)...);
    }
    m_condition_pop.notify_one();
    return true;
  }

  bool pop(T& item)
  {
    {
      std::unique_lock guard(m_queue_lock);
      m_condition_pop.wait(guard, [&]() { return !m_queue.empty() || !m_block; });
      if (m_queue.empty())
        return false;
      item = std::move(m_queue.front());
      m_queue.pop();
    }
    m_condition_push.notify_one();
    return true;
  }

  std::size_t size() const
  {
    std::scoped_lock guard(m_queue_lock);
    return m_queue.size();
  }

  std::size_t capacity() const { return m_max_size; }

  bool empty() const
  {
    std::scoped_lock guard(m_queue_lock);
    return m_queue.empty();
  }

  bool full() const
  {
    std::scoped_lock lock(m_queue_lock);
    return m_queue.size() == capacity();
  }

  void block()
  {
    std::scoped_lock guard(m_queue_lock);
    m_block = true;
  }

  void unblock()
  {
    {
      std::scoped_lock guard(m_queue_lock);
      m_block = false;
    }
    m_condition_push.notify_all();
    m_condition_pop.notify_all();
  }

  bool blocking() const
  {
    std::scoped_lock guard(m_queue_lock);
    return m_block;
  }

private:
  using queue_t = std::queue<T>;
  queue_t m_queue;

  bool m_block;
  const std::size_t m_max_size;

  mutable std::mutex m_queue_lock;
  std::condition_variable m_condition_push;
  std::condition_variable m_condition_pop;
};

class thread_pool
{
public:
  explicit thread_pool(std::size_t thread_count = std::thread::hardware_concurrency())
    : m_queues(thread_count)
    , m_count(thread_count)
  {
    if (!thread_count)
      throw std::invalid_argument("bad thread count! must be non-zero!");

    auto worker = [this](auto i) {
      while (true) {
        proc_t f;
        for (auto n = 0; n < m_count * K; ++n)
          if (m_queues[(i + n) % m_count].try_pop(f))
            break;
        if (!f && !m_queues[i].pop(f))
          break;
        f();
      }
    };

    m_threads.reserve(thread_count);
    for (auto i = 0; i < thread_count; ++i)
      m_threads.emplace_back(worker, i);
  }

  ~thread_pool()
  {
    for (auto& queue : m_queues)
      queue.unblock();
    for (auto& thread : m_threads)
      thread.join();
  }

  template<typename F, typename... Args>
  void enqueue_work(F&& f, Args&&... args)
  {
    auto work = [p = std::forward<F>(f), t = std::make_tuple(std::forward<Args>(args)...)]() { std::apply(p, t); };
    auto i = m_index++;

    for (auto n = 0; n < m_count * K; ++n)
      if (m_queues[(i + n) % m_count].try_push(work))
        return;

    m_queues[i % m_count].push(std::move(work));
  }

  template<typename F, typename... Args>
  [[nodiscard]] auto enqueue_task(F&& f, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>
  {
    using task_return_type = std::invoke_result_t<F, Args...>;
    using task_type = std::packaged_task<task_return_type()>;

    auto task = std::make_shared<task_type>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    auto work = [=]() { (*task)(); };
    auto result = task->get_future();
    auto i = m_index++;

    for (auto n = 0; n < m_count * K; ++n)
      if (m_queues[(i + n) % m_count].try_push(work))
        return result;

    m_queues[i % m_count].push(std::move(work));

    return result;
  }

private:
  using proc_t = std::function<void(void)>;
  using queue_t = unbounded_queue<proc_t>;
  using queues_t = std::vector<queue_t>;
  queues_t m_queues;

  using threads_t = std::vector<std::thread>;
  threads_t m_threads;

  const std::size_t m_count;
  std::atomic_uint m_index = 0;

  inline static const unsigned int K = 2;
};

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
