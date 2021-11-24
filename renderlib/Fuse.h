#pragma once

#include "glm.h"

#include <atomic>
#include <condition_variable>
#include <future>
#include <queue>
#include <thread>
#include <type_traits>
#include <vector>

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
  static const size_t NTHREADS;

  std::atomic_uint8_t m_nThreadsWorking = 0;
  std::vector<std::thread> m_threads;

  std::condition_variable m_conditionVar;
  std::mutex m_mutex;
  std::atomic_bool m_stop = false;

  // the read-only volume data
  const ImageXYZC* m_img;
  // the fused result (always rewrite to same mem location)
  uint8_t* m_outRGBVolume;

  // double-buffered so that one can be waiting while the other processing
  bounded_queue<std::vector<glm::vec3>> m_fuseRequests;
  //  std::queue<std::vector<glm::vec3>> m_fuseRequests;
  // std::vector<glm::vec3> m_colorsToFuse;
  // std::vector<glm::vec3> m_queuedColorsToFuse;

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
