#pragma once

#include <deque>
#include <functional>
#include <future>
#include <mutex>

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
parallel_for(size_t nb_elements, std::function<void(size_t start, size_t end)> functor, bool use_threads = true);

struct Tasks
{
  // the mutex, condition variable and deque form a single
  // thread-safe triggered queue of tasks:
  std::mutex m;
  std::condition_variable v;
  // note that a packaged_task<void> can store a packaged_task<R>:
  std::deque<std::packaged_task<bool()>> work;

  // this holds futures representing the worker threads being done:
  std::vector<std::future<void>> finished;

  // queue( lambda ) will enqueue the lambda into the tasks for the threads
  // to use.  A future of the type the lambda returns is given to let you get
  // the result out.
  template<class F, class R = std::result_of_t<F&()>>
  std::future<R> queue(F&& f);

  // start N threads in the thread pool.
  void start(std::size_t N = 1);

  // abort() cancels all non-started tasks, and tells every working thread
  // stop running, and waits for them to finish up.
  void abort();
  // cancel_pending() merely cancels all non-started tasks:
  void cancel_pending();
  // finish enques a "stop the thread" message for every thread, then waits for them:
  void finish();
  ~Tasks();

private:
  // the work that a worker thread does:
  void thread_task();
};
