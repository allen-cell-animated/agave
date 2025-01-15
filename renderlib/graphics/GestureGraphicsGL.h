#pragma once

#include "glad/glad.h"

#include "graphics/glsl/GLGuiShader.h"
#include "graphics/glsl/GLThickLines.h"
#include "graphics/gl/FontGL.h"
#include "gesture/gesture.h"

#include <memory>

// a vertex buffer that is automatically allocated and then deleted when it goes out of scope
class ScopedGlVertexBuffer
{
public:
  ScopedGlVertexBuffer(const void* data, size_t size);
  ~ScopedGlVertexBuffer();
  GLuint buffer() const { return m_buffer; }

private:
  GLuint m_vertexArray;
  GLuint m_buffer;
};

// a vertex buffer that is automatically allocated and then deleted when it goes out of scope
class ScopedGlTextureBuffer
{
public:
  ScopedGlTextureBuffer(const void* data, size_t size);
  ~ScopedGlTextureBuffer();
  GLuint buffer() const { return m_buffer; }
  GLuint texture() const { return m_texture; }

private:
  GLuint m_texture;
  GLuint m_buffer;
};

// Some base RenderBuffer struct, in common between viewport rendering and
// other stuff...
// TODO reconcile with Graphics/FrameBuffer.h
struct RenderBuffer
{
  glm::ivec2 resolution = glm::ivec2(0);
  int samples = 0;
  uint32_t frameBuffer = 0;
  uint32_t depthRenderBuffer = 0;
  uint32_t renderedTexture = 0;
  uint32_t depthTexture = 0;

  // Call update at the beginning of scene/frame update. We need to make sure
  // we do have a frame buffer (created lazily) at the right resolution.
  // @param samples is for MSAA, should be zero for selection buffers.
  bool update(glm::ivec2 resolution, int samples = 0)
  {
    if (resolution == this->resolution && samples == this->samples)
      return true;

    // To prevent some buffer allocation issue, ignore zero-size buffer resize, this happens when
    // the app is minimized.
    if (resolution == glm::ivec2(0)) {
      return true;
    }

    destroy();
    bool ok = create(resolution, samples);
    if (ok) {
      clear();
    }
    return ok;
  }
  void destroy();
  bool create(glm::ivec2 resolution, int samples = 0);
  virtual void clear() {};
};

struct SelectionBuffer : RenderBuffer
{
  virtual void clear();
};

class GestureRendererGL
{
public:
  std::unique_ptr<GLGuiShader> shader;
  std::unique_ptr<GLThickLinesShader> shaderLines;

  // A texture atlas for GUI elements
  // TODO: use bindless textures
  uint32_t glTextureId = 0;

  std::unique_ptr<FontGL> font;

  // Gesture draw, called once per window update (frame) when the GUI draw commands
  // had been described in full.
  void draw(struct SceneView& sceneView, struct SelectionBuffer* selection, Gesture::Graphics& graphics);

  // Pick a GUI element using the cursor position in Input.
  // Return a valid GUI selection code, Gesture::Graphics::k_noSelectionCode
  // otherwise.
  // viewport: left, top, width, height
  bool pick(struct SelectionBuffer& selection,
            const Gesture::Input& input,
            const SceneView::Viewport& viewport,
            uint32_t& selectionCode);
};
