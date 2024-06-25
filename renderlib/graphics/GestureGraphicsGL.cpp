#include "GestureGraphicsGL.h"

#include "graphics/gl/FontGL.h"
#include "graphics/gl/Util.h"
#include "graphics/glsl/GLGuiShader.h"

// a vertex buffer that is automatically allocated and then deleted when it goes out of scope
ScopedGlVertexBuffer::ScopedGlVertexBuffer(const void* data, size_t size)
{
  glGenVertexArrays(1, &m_vertexArray);
  glBindVertexArray(m_vertexArray);

  glGenBuffers(1, &m_buffer);
  glBindBuffer(GL_ARRAY_BUFFER, m_buffer);
  glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);

  const size_t vtxStride = 9 * sizeof(GLfloat) + 1 * sizeof(GLuint);

  // xyz uv rgba s

  // specify position attribute
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vtxStride, (GLvoid*)0);
  glEnableVertexAttribArray(0); // m_loc_vpos

  // specify uv attribute
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, vtxStride, (GLvoid*)(3 * sizeof(GLfloat)));
  glEnableVertexAttribArray(1); // m_loc_vuv

  // specify color rgba attribute
  glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, vtxStride, (GLvoid*)(5 * sizeof(GLfloat)));
  glEnableVertexAttribArray(2); // m_loc_vcolor

  // specify selection id attribute
  glVertexAttribIPointer(3, 1, GL_UNSIGNED_INT, vtxStride, (GLvoid*)(9 * sizeof(GLfloat)));
  glEnableVertexAttribArray(3); // m_loc_vcode
}
ScopedGlVertexBuffer::~ScopedGlVertexBuffer()
{
  glBindVertexArray(0);
  glDeleteVertexArrays(1, &m_vertexArray);
  glDeleteBuffers(1, &m_buffer);
}

namespace Pipeline {

// First I may draw any GUI geometry that I want to be depth-composited with the
// rest of the scene in viewport. This may be any supporting guide that needs to
// appear for the duration of some action and reveal intersections against the
// scene geometry.

// Draw something "in the scene". This has a limitation that we assume there is a
// single viewport.
static void
configure_3dDepthTested(SceneView& sceneView, Gesture::Graphics& graphics)
{
  auto& shader = graphics.shader;

  glm::mat4 v(1.0);
  sceneView.camera.getViewMatrix(v);
  glm::mat4 p(1.0);
  sceneView.camera.getProjMatrix(p);

  glUniformMatrix4fv(shader->m_loc_proj, 1, GL_FALSE, glm::value_ptr(p * v));
  check_gl("set proj matrix");

  glEnable(GL_DEPTH_TEST);
  check_gl("enable depth test");
}

// The second pass is still about 3d geometry, only this time I want it to be
// drawn on top, without depth test. These two passes shares in common the same
// projection matrix as the rest of the scene. 3d manipulators shown earlier are examples.

// Overlay something "in the scene". This has a limitation that we assume there
// is a single viewport.
static void
configure_3dStacked(SceneView& sceneView, Gesture::Graphics& graphics)
{
  auto& shader = graphics.shader;

  glm::mat4 v(1.0);
  sceneView.camera.getViewMatrix(v);
  glm::mat4 p(1.0);
  sceneView.camera.getProjMatrix(p);
  check_gl("PRE set proj matrix");

  glUniformMatrix4fv(shader->m_loc_proj, 1, GL_FALSE, glm::value_ptr(p * v));

  check_gl("set proj matrix");

  glDisable(GL_DEPTH_TEST);
  check_gl("disable depth test");
}

// The third pass is a 2d orthographic projection of screen space, where the
// coordinates are measured in pixels starting at the lower left corner of the
// screen. Here is where I draw buttons or other traditional GUI elements if you wish.

// Draw something in screen space without zbuffer.
static void
configure_2dScreen(SceneView& sceneView, Gesture::Graphics& graphics)
{
  auto& shader = graphics.shader;

  auto p = glm::ortho((float)sceneView.viewport.region.lower.x,
                      (float)sceneView.viewport.region.upper.x,
                      (float)sceneView.viewport.region.lower.y,
                      (float)sceneView.viewport.region.upper.y,
                      1.0f,
                      -1.f);
  glUniformMatrix4fv(shader->m_loc_proj, 1, GL_FALSE, glm::value_ptr(p));
  check_gl("set proj matrix");

  glDisable(GL_DEPTH_TEST);
  check_gl("disable depth test");
}
} // namespace Pipeline

template<typename DrawBlock>
void
drawGestureCodes(const Gesture::Graphics::SelectionBuffer& selection,
                 const SceneView::Viewport& viewport,
                 DrawBlock drawSceneGeometry)
{
  // Backup
  GLenum last_framebuffer;
  glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, (GLint*)&last_framebuffer);
  check_gl("get draw framebuffer");
  GLboolean last_enable_depth_test = glIsEnabled(GL_DEPTH_TEST);
  check_gl("is depth enabled");
  GLboolean last_enable_blend = glIsEnabled(GL_BLEND);
  check_gl("is blend enabled");
  GLfloat last_clear_color[4];
  glGetFloatv(GL_COLOR_CLEAR_VALUE, last_clear_color);

  // Render to texture
  glBindFramebuffer(GL_FRAMEBUFFER, selection.frameBuffer);
  check_gl("bind selection framebuffer");
  {
    glViewport(viewport.region.lower.x, viewport.region.lower.y, viewport.region.upper.x, viewport.region.upper.y);
    glDisable(GL_BLEND);
    uint32_t clearcode = Gesture::Graphics::SelectionBuffer::k_noSelectionCode;
    glClearColor(((clearcode >> 0) & 0xFF) / 255.0,
                 ((clearcode >> 8) & 0xFF) / 255.0,
                 ((clearcode >> 16) & 0xFF) / 255.0,
                 ((clearcode >> 24) & 0xFF) / 255.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    drawSceneGeometry();
  }

  // Restore
  glBindFramebuffer(GL_FRAMEBUFFER, last_framebuffer);
  check_gl("restore default framebuffer");
  if (last_enable_depth_test) {
    glEnable(GL_DEPTH_TEST);
  } else {
    glDisable(GL_DEPTH_TEST);
  }
  check_gl("restore depth test state");
  if (last_enable_blend) {
    glEnable(GL_BLEND);
  } else {
    glDisable(GL_BLEND);
  }
  check_gl("restore blend enabled state");
  glClearColor(last_clear_color[0], last_clear_color[1], last_clear_color[2], last_clear_color[3]);
  check_gl("restore clear color");
}
void
Gesture::Graphics::RenderBuffer::destroy()
{
  if (frameBuffer == 0) {
    return;
  }
  glDeleteFramebuffers(1, &frameBuffer);
  glDeleteRenderbuffers(1, &depthRenderBuffer);
  glDeleteTextures(1, &renderedTexture);
  glDeleteTextures(1, &depthTexture);
  frameBuffer = 0;
  depthRenderBuffer = 0;
  renderedTexture = 0;
  resolution = glm::ivec2(0, 0);
}

bool
Gesture::Graphics::RenderBuffer::create(glm::ivec2 resolution, int samples)
{
  this->resolution = resolution;
  this->samples = samples;

  GLint last_framebuffer;
  glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &last_framebuffer);

  glGenFramebuffers(1, &frameBuffer);
  glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);

  if (samples == 0) {
    glGenTextures(1, &renderedTexture);
    // glCreateTextures(GL_TEXTURE_2D, 1, texturePtr);

    // "Bind" the newly created texture: all future texture functions will modify this texture
    glBindTexture(GL_TEXTURE_2D, renderedTexture);

    // Define the texture quality and zeroes its memory
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, resolution.x, resolution.y, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

    // We don't need texture filtering, but we need to specify some.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    // Set "renderedTexture" as our colour attachement #0
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderedTexture, 0);

    // The depth buffer
    glGenRenderbuffers(1, &depthRenderBuffer);
    glBindRenderbuffer(GL_RENDERBUFFER, depthRenderBuffer);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, resolution.x, resolution.y);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderBuffer);
  } else {
    glGenTextures(1, &renderedTexture);
    glGenTextures(1, &depthTexture);
    // glCreateTextures(GL_TEXTURE_2D_MULTISAMPLE, 1, &renderedTexture);
    // glCreateTextures(GL_TEXTURE_2D_MULTISAMPLE, 1, &depthTexture);

    // "Bind" the newly created texture : all future texture functions will modify this texture
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, renderedTexture);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, samples, GL_RGBA, resolution.x, resolution.y, GL_TRUE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, renderedTexture, 0);

    // The depth buffer
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, depthTexture);
    glTexImage2DMultisample(
      GL_TEXTURE_2D_MULTISAMPLE, samples, GL_DEPTH32F_STENCIL8, resolution.x, resolution.y, GL_TRUE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D_MULTISAMPLE, depthTexture, 0);
  }

  // Always check that our framebuffer is ok
  bool status = (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
  check_glfb("renderbuffer for picking");

  glBindFramebuffer(GL_FRAMEBUFFER, last_framebuffer);
  return status;
}

void
Gesture::Graphics::SelectionBuffer::clear()
{
  // Backup
  GLenum last_framebuffer;
  glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, (GLint*)&last_framebuffer);
  GLfloat last_clear_color[4];
  glGetFloatv(GL_COLOR_CLEAR_VALUE, last_clear_color);

  // Render to texture
  glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
  {
    glViewport(0, 0, resolution.x, resolution.y);
    uint32_t clearcode = Gesture::Graphics::SelectionBuffer::k_noSelectionCode;
    glClearColor(((clearcode >> 0) & 0xFF) / 255.0,
                 ((clearcode >> 8) & 0xFF) / 255.0,
                 ((clearcode >> 16) & 0xFF) / 255.0,
                 ((clearcode >> 24) & 0xFF) / 255.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  }

  // Restore
  glBindFramebuffer(GL_FRAMEBUFFER, last_framebuffer);
  glClearColor(last_clear_color[0], last_clear_color[1], last_clear_color[2], last_clear_color[3]);
}
