#include "RenderGL.h"

#include "glad/glad.h"

#include "ImageXYZC.h"
#include "Logging.h"
#include "RenderSettings.h"
#include "gl/v33/V33Image3D.h"

#include <iostream>

RenderGL::RenderGL(RenderSettings* rs)
  : m_image3d(nullptr)
  , m_w(0)
  , m_h(0)
  , m_renderSettings(rs)
  , m_scene(nullptr)
{}

RenderGL::~RenderGL()
{
  delete m_image3d;
}

void
RenderGL::initialize(uint32_t w, uint32_t h)
{
  GLint max_combined_texture_image_units;
  glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &max_combined_texture_image_units);
  LOG_DEBUG << "GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS: " << max_combined_texture_image_units;

  glEnable(GL_DEPTH_TEST);
  glEnable(GL_CULL_FACE);
  // glEnable(GL_MULTISAMPLE);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  if (m_scene && m_scene->m_volume) {
    initFromScene();
  }

  // Size viewport
  resize(w, h);
}

void
RenderGL::render(const CCamera& camera)
{
  if (!m_scene || !m_scene->m_volume) {
    return;
  }
  if (!m_image3d) {
    initFromScene();
  }

  glClearColor(0.0, 0.0, 0.0, 1.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  if (!m_image3d) {
    return;
  }

  if (m_renderSettings->m_DirtyFlags.HasFlag(RenderParamsDirty | TransferFunctionDirty | VolumeDataDirty)) {
    m_image3d->prepareTexture(*m_scene);
  }

  // At this point, all dirty flags should have been taken care of, since the flags in the original scene are now
  // cleared
  m_renderSettings->m_DirtyFlags.ClearAllFlags();

  // Render image
  m_image3d->render(camera, m_scene, m_renderSettings);

  m_timingRender.AddDuration((float)m_timer.elapsed());
  m_status.SetStatisticChanged(
    "Performance", "Render Image", QString::number(m_timingRender.m_FilteredDuration, 'f', 2), "ms.");
  m_timer.start();
}

void
RenderGL::resize(uint32_t w, uint32_t h)
{
  m_w = w;
  m_h = h;
  glViewport(0, 0, w, h);
}

RenderParams&
RenderGL::renderParams()
{
  return m_renderParams;
}
Scene*
RenderGL::scene()
{
  return m_scene;
}
void
RenderGL::setScene(Scene* s)
{
  m_scene = s;
}

void
RenderGL::cleanUpResources()
{
  delete m_image3d;
  m_image3d = nullptr;
}

void
RenderGL::initFromScene()
{
  delete m_image3d;

  m_image3d = new Image3Dv33(m_scene->m_volume);
  m_image3d->create();

  // we have set up everything there is to do before rendering
  m_timer.start();
  m_status.SetRenderBegin();
}
