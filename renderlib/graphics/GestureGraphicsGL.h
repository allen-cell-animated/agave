#pragma once

///////////////////////////////////////////////

// 1. make a gesture graphics renderer that accepts a Gesture::Graphics object
// and depends on Gesture::Graphics, not the other way around. Gesture::Graphics should not depend on any GL.

// 2. factor gl stuff out of Font so that Font just wraps the glyph data but not the texture

// 3. the ViewerWindow uses the GestureRendererGL to draw the GUI elements just like it uses a RenderGLPT to draw the
// scene

///////////////////////////////////////////////

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
  virtual void clear(){};
};

struct SelectionBuffer : RenderBuffer
{
  // 1 bit is reserved for component flags.
  static constexpr uint32_t k_noSelectionCode = 0x7fffffffu;

  virtual void clear();
};

class GestureRendererGL
{
public:
  std::unique_ptr<GLGuiShader> shader;

  // A texture atlas for GUI elements
  // TODO: use bindless textures
  uint32_t glTextureId = 0;

  std::unique_ptr<FontGL> font;

  // Gesture draw, called once per window update (frame) when the GUI draw commands
  // had been described in full.
  void draw(struct SceneView& sceneView, struct SelectionBuffer* selection);

  // Pick a GUI element using the cursor position in Input.
  // Return a valid GUI selection code, SelectionBuffer::k_noSelectionCode
  // otherwise.
  // viewport: left, top, width, height
  bool pick(struct SelectionBuffer& selection, const Input& input, const SceneView::Viewport& viewport);

  // the one "scene" that this class will render
  Gesture::Graphics* graphics;
};
