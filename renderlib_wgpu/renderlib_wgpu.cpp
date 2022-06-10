#include "renderlib_wgpu.h"

#include "../renderlib/Logging.h"

#include <string>

static bool renderLibInitialized = false;

static bool renderLibHeadless = false;

static const uint32_t AICS_DEFAULT_STENCIL_BUFFER_BITS = 8;

static const uint32_t AICS_DEFAULT_DEPTH_BUFFER_BITS = 24;

int
renderlib_wgpu::initialize(bool headless, bool listDevices, int selectedGpu)
{
  if (renderLibInitialized) {
    return 1;
  }
  renderLibInitialized = true;

  renderLibHeadless = headless;

  LOG_INFO << "Renderlib_wgpu startup";

  bool enableDebug = false;

  if (headless) {
  } else {
  }

  if (enableDebug) {
  }

  // load gl functions and init stuff

  // then log out some info
  // LOG_INFO << "GL_VENDOR: " << std::string((char*)glGetString(GL_VENDOR));
  // LOG_INFO << "GL_RENDERER: " << std::string((char*)glGetString(GL_RENDERER));

  return 0;
}

void
renderlib_wgpu::cleanup()
{
  if (!renderLibInitialized) {
    return;
  }
  LOG_INFO << "Renderlib_wgpu shutdown";

  if (renderLibHeadless) {
  }
  renderLibInitialized = false;
}
