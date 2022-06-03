#pragma once

#include <vulkan/vulkan.h>

#include "../renderlib/Logging.h"

// we want to immediately abort when there is an error.
// In normal engines this would give an error message to the user, or perform a dump of state.
#define VK_CHECK(x)                                                                                                    \
  do {                                                                                                                 \
    VkResult err = x;                                                                                                  \
    if (err) {                                                                                                         \
      LOG_ERROR << "Detected Vulkan error: " << err;                                                                   \
    }                                                                                                                  \
  } while (0)
