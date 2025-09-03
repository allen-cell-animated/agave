#include "RenderWgpuPT.h"

#include "../renderlib/Status.h"

RenderWgpuPT::RenderWgpuPT(RenderSettings* rs)
  : m_scene(nullptr)
  , m_w(0)
  , m_h(0)
  , m_renderSettings(rs)
  , m_status(new CStatus)
{
}
RenderWgpuPT::~RenderWgpuPT() {}

void
RenderWgpuPT::initialize(uint32_t w, uint32_t h)
{
}
void
RenderWgpuPT::render(const CCamera& camera)
{
}
void
RenderWgpuPT::renderTo(const CCamera& camera, IRenderTarget* fbo)
{
}
void
RenderWgpuPT::resize(uint32_t w, uint32_t h)
{
}
