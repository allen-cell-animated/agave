#pragma once

#include "vk.h"

#include <vector>

template<class TValue>
inline void
GetEnumerateVector(VkResult(enumerate)(uint32_t*, TValue*), std::vector<TValue>& vector)
{
  uint32_t count = 0;
  VK_CHECK(enumerate(&count, nullptr));

  vector.resize(count);
  VK_CHECK(enumerate(&count, vector.data()));
}

template<class THandle, class TValue>
inline void
GetEnumerateVector(THandle handle, void(enumerate)(THandle, uint32_t*, TValue*), std::vector<TValue>& vector)
{
  uint32_t count = 0;
  enumerate(handle, &count, nullptr);

  vector.resize(count);
  enumerate(handle, &count, vector.data());
}

template<class THandle, class TValue>
inline void
GetEnumerateVector(THandle handle, VkResult(enumerate)(THandle, uint32_t*, TValue*), std::vector<TValue>& vector)
{
  uint32_t count = 0;
  VK_CHECK(enumerate(handle, &count, nullptr));

  vector.resize(count);
  VK_CHECK(enumerate(handle, &count, vector.data()));
}

template<class THandle1, class THandle2, class TValue>
inline void
GetEnumerateVector(THandle1 handle1,
                   THandle2 handle2,
                   VkResult(enumerate)(THandle1, THandle2, uint32_t*, TValue*),
                   std::vector<TValue>& vector)
{
  uint32_t count = 0;
  VK_CHECK(enumerate(handle1, handle2, &count, nullptr));

  vector.resize(count);
  VK_CHECK(enumerate(handle1, handle2, &count, vector.data()));
}

template<class TValue>
inline std::vector<TValue>
GetEnumerateVector(VkResult(enumerate)(uint32_t*, TValue*))
{
  std::vector<TValue> initial;
  GetEnumerateVector(enumerate, initial);
  return initial;
}

template<class THandle, class TValue>
inline std::vector<TValue>
GetEnumerateVector(THandle handle, void(enumerate)(THandle, uint32_t*, TValue*))
{
  std::vector<TValue> initial;
  GetEnumerateVector(handle, enumerate, initial);
  return initial;
}

template<class THandle, class TValue>
inline std::vector<TValue>
GetEnumerateVector(THandle handle, VkResult(enumerate)(THandle, uint32_t*, TValue*))
{
  std::vector<TValue> initial;
  GetEnumerateVector(handle, enumerate, initial);
  return initial;
}

template<class THandle1, class THandle2, class TValue>
inline std::vector<TValue>
GetEnumerateVector(THandle1 handle1, THandle2 handle2, VkResult(enumerate)(THandle1, THandle2, uint32_t*, TValue*))
{
  std::vector<TValue> initial;
  GetEnumerateVector(handle1, handle2, enumerate, initial);
  return initial;
}
