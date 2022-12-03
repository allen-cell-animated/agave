#include "threading.h"

#include <algorithm>
#include <thread>
#include <vector>

/// @param[in] nb_elements : size of your for loop
/// @param[in] functor(start, end) :
/// your function processing a sub chunk of the for loop.
/// "start" is the first index to process (included) until the index "end"
/// (excluded)
/// @code
///     for(int i = start; i < end; ++i)
///         computation(i);
/// @endcode
/// @param use_threads : enable / disable threads.
///
///
void
parallel_for(size_t nb_elements, std::function<void(size_t start, size_t end)> functor, bool use_threads)
{
  // -------
  unsigned nb_threads_hint = std::thread::hardware_concurrency();
  unsigned nb_threads = nb_threads_hint == 0 ? 8 : (nb_threads_hint);

  size_t batch_size = nb_elements / nb_threads;
  size_t batch_remainder = nb_elements % nb_threads;

  std::vector<std::thread> my_threads(nb_threads);

  if (use_threads) {
    // Multithread execution
    for (size_t i = 0; i < nb_threads; ++i) {
      size_t start = i * batch_size;
      my_threads[i] = std::thread(functor, start, start + batch_size);
    }
  } else {
    // Single thread execution (for easy debugging)
    for (size_t i = 0; i < nb_threads; ++i) {
      size_t start = i * batch_size;
      functor(start, start + batch_size);
    }
  }

  // Deform the elements left, on THIS thread
  size_t start = nb_threads * batch_size;
  functor(start, start + batch_remainder);

  // Wait for the other thread to finish their task
  if (use_threads)
    std::for_each(my_threads.begin(), my_threads.end(), std::mem_fn(&std::thread::join));
}

// queue( lambda ) will enqueue the lambda into the tasks for the threads
// to use.  A future of the type the lambda returns is given to let you get
// the result out.
template<class F, class R>
std::future<R>
Tasks::queue(F&& f)
{
  // wrap the function object into a packaged task, splitting
  // execution from the return value:
  std::packaged_task<R()> p(std::forward<F>(f));

  auto r = p.get_future(); // get the return value before we hand off the task
  {
    std::unique_lock<std::mutex> l(m);
    work.emplace_back(std::move(p)); // store the task<R()> as a task<void()>
  }
  v.notify_one(); // wake a thread to work on the task

  return r; // return the future result of the task
}

// start N threads in the thread pool.
void
Tasks::start(std::size_t N)
{
  for (std::size_t i = 0; i < N; ++i) {
    // each thread is a std::async running this->thread_task():
    finished.push_back(std::async(std::launch::async, [this] { thread_task(); }));
  }
}

// abort() cancels all non-started tasks, and tells every working thread
// stop running, and waits for them to finish up.
void
Tasks::abort()
{
  cancel_pending();
  finish();
}
// cancel_pending() merely cancels all non-started tasks:
void
Tasks::cancel_pending()
{
  std::unique_lock<std::mutex> l(m);
  work.clear();
}
// finish enques a "stop the thread" message for every thread, then waits for them:
void
Tasks::finish()
{
  {
    std::unique_lock<std::mutex> l(m);
    for (auto&& unused : finished) {
      work.push_back({});
    }
  }
  v.notify_all();
  finished.clear();
}
Tasks::~Tasks()
{
  finish();
}

// the work that a worker thread does:
void
Tasks::thread_task()
{
  while (true) {
    // pop a task off the queue:
    std::packaged_task<bool()> f;
    {
      // usual thread-safe queue code:
      std::unique_lock<std::mutex> l(m);
      if (work.empty()) {
        v.wait(l, [&] { return !work.empty(); });
      }
      f = std::move(work.front());
      work.pop_front();
    }
    // if the task is invalid, it means we are asked to abort:
    if (!f.valid())
      return;
    // otherwise, run the task:
    f();
  }
}

#if 0
// usage example:
unsigned int min_cores = 1; // for the case when hardware_concurency fails and returns 0
unsigned int number_of_cores = std::max(min_cores, std::min(cores_, std::thread::hardware_concurrency() - 1));
{
  std::vector<std::future<bool>> jobs;
  Tasks tasks;
  for_each(matches.begin(), matches.end(), [&](const SubblockIndexVec::value_type& match_) {
    jobs.push_back(tasks.queue([]() -> bool {
      return true;
    }));

  });
  tasks.start(number_of_cores);
  for_each(jobs.begin(), jobs.end(), [](auto& x) { x.get(); });
}
#endif