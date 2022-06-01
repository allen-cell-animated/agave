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
parallel_for(size_t nb_elements, std::function<void(int start, int end)> functor, bool use_threads)
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
