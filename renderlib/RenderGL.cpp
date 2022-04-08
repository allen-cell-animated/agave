#include "RenderGL.h"

#include "glad/glad.h"

#include "ImageXYZC.h"
#include "Logging.h"
#include "RenderSettings.h"
#include "gl/V33Image3D.h"

#include <iostream>

RenderGL::RenderGL(RenderSettings* rs)
  : m_image3d(nullptr)
  , m_w(0)
  , m_h(0)
  , m_renderSettings(rs)
  , m_scene(nullptr)
  , m_devicePixelRatio(1.0f)
  , m_status(new CStatus)
{
  mStartTime = std::chrono::high_resolution_clock::now();
}

RenderGL::~RenderGL()
{
  delete m_image3d;
}

void
RenderGL::initialize(uint32_t w, uint32_t h, float devicePixelRatio)
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
  resize(w, h, devicePixelRatio);
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

  if (m_scene) {
    glClearColor(m_scene->m_material.m_backgroundColor[0],
                 m_scene->m_material.m_backgroundColor[1],
                 m_scene->m_material.m_backgroundColor[2],
                 1.0);
  } else {
    glClearColor(0.0, 0.0, 0.0, 1.0);
  }
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

  glViewport(0, 0, (GLsizei)(m_w * m_devicePixelRatio), (GLsizei)(m_h * m_devicePixelRatio));
  // Render image
  m_image3d->render(camera, m_scene, m_renderSettings, m_devicePixelRatio);

  auto endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = endTime - mStartTime;
  m_timingRender.AddDuration((float)(elapsed.count() * 1000.0));
  m_status->SetStatisticChanged("Performance", "Render Image", m_timingRender.filteredDurationAsString(), "ms.");
  mStartTime = std::chrono::high_resolution_clock::now();
}

void
RenderGL::resize(uint32_t w, uint32_t h, float devicePixelRatio)
{
  m_w = w;
  m_h = h;
  m_devicePixelRatio = devicePixelRatio;
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
  mStartTime = std::chrono::high_resolution_clock::now();
  m_status->SetRenderBegin();
}
