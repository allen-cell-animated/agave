#include "gesture.h"
#include "gl/Util.h"
#include "graphics/glsl/GLGuiShader.h"

// Update the current action for one of the button of the pointer device
void
Gesture::Input::setButtonEvent(uint32_t mbIndex, Action action, int mods, glm::vec2 position, double time)
{
  if (mbIndex >= kButtonsCount) {
    return;
  }

  Button& button = mbs[mbIndex];
  if (action == Input::kPress) {
    // If the the button is pressed and was previously in a neutral state, it could be a click or a
    // double-click.
    if (button.action == Gesture::Input::kNone) {
      // A double-click event is recorded if the new click follows a previous click of the
      // same button within some time interval.
      button.doubleClick = (time - button.triggerTime) < doubleClickTime;
      button.triggerTime = time;

      // Record position of the pointer during the press event, we are going to use it to
      // determine drag operations
      button.pressedPosition = position;

      // The state of modifiers are recorded when buttons are initially pressed. The state
      // is retained for the duration of the click/drag. We do not update this state until the
      // next button press event.
      button.modifier = mods;
    }

    button.action = Gesture::Input::kPress;
  } else if (action == Input::kRelease) {
    // When button is released, record any drag distance
    if (button.action != Gesture::Input::kNone) {
      glm::vec2 origin = button.pressedPosition;
      button.drag = position - origin;
    }

    button.action = Gesture::Input::kRelease;
  }
}

void
Gesture::Input::setPointerPosition(glm::vec2 position)
{
  cursorPos = position;

  // Update each button action. Each button holding a kPress event becomes a
  // kDrag event.
  for (int mbIndex = 0; mbIndex < Input::kButtonsCount; ++mbIndex) {
    Button& button = mbs[mbIndex];
    if (button.action == Input::kNone || button.action == Input::kRelease) {
      continue;
    }

    glm::vec2 origin = button.pressedPosition;
    glm::vec2 drag = position - origin;
    button.drag = drag;
    bool anyMotion = drag != glm::vec2(0);

    if (button.action == Gesture::Input::kPress && anyMotion) {
      button.action = Gesture::Input::kDrag;

      // If we hold the shift modifier we record if the initial drag is mostly
      // horizontal or vertical. This information may be used by some tools.
      if (button.modifier & Gesture::Input::kShift) {
        button.dragConstraint = (abs(drag.x) > abs(drag.y) ? Gesture::Input::kHorizontal : Gesture::Input::kVertical);
      } else {
        button.dragConstraint = Gesture::Input::kUnconstrained;
      }
    }
  }
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

// a vertex buffer that is automatically allocated and then deleted when it goes out of scope
class ScopedGlVertexBuffer
{
public:
  ScopedGlVertexBuffer(const void* data, size_t size)
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
  ~ScopedGlVertexBuffer()
  {
    glBindVertexArray(0);
    glDeleteVertexArrays(1, &m_vertexArray);
    glDeleteBuffers(1, &m_buffer);
  }
  GLuint buffer() const { return m_buffer; }

private:
  GLuint m_vertexArray;
  GLuint m_buffer;
};

void
Gesture::Graphics::draw(SceneView& sceneView, const SelectionBuffer* selection)
{
  // Gesture draw spans across the entire window and it is not restricted to a single
  // viewport.
  if (this->verts.empty()) {
    clearCommands();
    return;
  }

  // lazy init
  if (!shader.get()) {
    shader.reset(new GLGuiShader());
  }

  // YAGNI: With a small effort we could create dynamic passes that are
  //        fully user configurable...
  //
  // Configure command lists
  void (*pipelineConfig[3])(SceneView&, Graphics&);
  // Step 1: we draw any command that is depth-composited with the scene
  pipelineConfig[static_cast<int>(Graphics::CommandSequence::k3dDepthTested)] = Pipeline::configure_3dDepthTested;
  // Step 2: we draw any command that is not depth composited but is otherwise using
  //         the same perspective projection
  pipelineConfig[static_cast<int>(Graphics::CommandSequence::k3dStacked)] = Pipeline::configure_3dStacked;
  // Step 3: we draw anything that is just an overlay in creen space. Most of the UI
  //         elements go here.
  pipelineConfig[static_cast<int>(Graphics::CommandSequence::k2dScreen)] = Pipeline::configure_2dScreen;

  // Backup state
  float lineWidth;
  glGetFloatv(GL_LINE_WIDTH, &lineWidth);
  check_gl("get line width");
  float pointSize;
  glGetFloatv(GL_POINT_SIZE, &pointSize);
  check_gl("get point size");
  bool depthTest = glIsEnabled(GL_DEPTH_TEST);
  check_gl("is depth test enabled");

  glEnable(GL_CULL_FACE);

  // Draw UI and viewport manipulators
  {
    // TODO are we really creating, uploading, and destroying the vertex buffer every frame?
    ScopedGlVertexBuffer vertex_buffer(this->verts.data(), this->verts.size() * sizeof(VertsCode));

    // Prepare a lambda to draw the Gesture commands. We'll run the lambda twice, once to
    // draw the GUI and once to draw the selection buffer data.
    // (display var is for draw vs pick)
    auto drawGesture = [&](bool display) {
      shader->configure(display, this->glTextureId);

      for (int sequence = 0; sequence < Graphics::kNumCommandsLists; ++sequence) {
        if (!this->commands[sequence].empty()) {
          pipelineConfig[sequence](sceneView, *this);

          // YAGNI: Commands could be coalesced, setting state could be avoided
          //        if not changing... For now it seems we can draw at over 2000 Hz
          //        and no further optimization is required.
          for (Graphics::CommandRange cmdr : this->commands[sequence]) {
            Graphics::Command& cmd = cmdr.command;
            if (cmdr.end == -1)
              cmdr.end = this->verts.size();
            if (cmdr.begin >= cmdr.end)
              continue;

            if (cmd.command == GL_LINES) {
              glLineWidth(cmd.thickness);
              check_gl("linewidth");
            }
            if (cmd.command == GL_POINTS) {
              glPointSize(cmd.thickness);
              check_gl("pointsize");
            }

            glDrawArrays(cmd.command, cmdr.begin, cmdr.end - cmdr.begin);
            check_gl("drawarrays");
          }
        }
      }

      shader->cleanup();
      check_gl("disablevertexattribarray");
    };
    drawGesture(/*display*/ true);

    // The last thing we draw is selection codes for next frame. This allows us
    // to know what is under the pointer cursor.
    if (selection) {
      drawGestureCodes(*selection, sceneView.viewport, [&]() { drawGesture(/*display*/ false); });
    }
  }

  // Restore state
  glLineWidth(lineWidth);
  check_gl("linewidth");
  glPointSize(pointSize);
  check_gl("pointsize");
  if (depthTest) {
    glEnable(GL_DEPTH_TEST);
  } else {
    glDisable(GL_DEPTH_TEST);
  }
  check_gl("toggle depth test");

  clearCommands();
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

SceneView::Viewport::Region
SceneView::Viewport::Region::intersect(const SceneView::Viewport::Region& a, const SceneView::Viewport::Region& b)
{
  Region r;
  r.lower.x = std::max(a.lower.x, b.lower.x);
  r.lower.y = std::max(a.lower.y, b.lower.y);
  r.upper.x = std::min(a.upper.x, b.upper.x);
  r.upper.y = std::min(a.upper.y, b.upper.y);
  return r;
}

uint32_t
selectionRGB8ToCode(const uint8_t* rgba)
{
  // ignores 4th component (== 0)
  uint32_t code = (uint32_t(rgba[0]) << 0) | (uint32_t(rgba[1]) << 8) | (uint32_t(rgba[2]) << 16);
  return code == 0xffffff ? Gesture::Graphics::SelectionBuffer::k_noSelectionCode : code;
}

bool
Gesture::Graphics::pick(SelectionBuffer& selection, const Gesture::Input& input, const SceneView::Viewport& viewport)
{
  // If we are in mid-gesture, then we can continue to use the retained selection code.
  // if we never had anything to draw into the pick buffer, then we didn't pick anything.
  // This is a slight oversimplification because it checks for any single-button release
  // or drag: two-button drag gestures are not handled well.
  if (input.clickEnded() || input.isDragging()) {
    return m_retainedSelectionCode != SelectionBuffer::k_noSelectionCode;
  }

  // Prepare a region in raster space

  SceneView::Viewport::Region region;
  {
    glm::ivec2 pixel = viewport.toRaster(input.cursorPos);

    // Grow the click position by some pixels to improve usability. Ideally
    // this should be a configurable parameter to improve accessibility.
    constexpr int kClickRadius = 7; //< in pixels
    region.extend(pixel - glm::ivec2(kClickRadius));
    region.extend(pixel + glm::ivec2(kClickRadius));
  }

  // Render on the whole framebuffer, complete from the lower left corner to the upper right
  SceneView::Viewport::Region viewRegion(viewport.region.lower, viewport.region.upper - glm::ivec2(1));

  // Crop selection with view in order to feed GL draw a valid region.
  region = SceneView::Viewport::Region::intersect(region, viewRegion);

  // if the intersection is empty, return no selection
  if (region.empty()) {
    m_retainedSelectionCode = SelectionBuffer::k_noSelectionCode;
    return false;
  }

  // Frame buffer resolution should be correct, check just in case.
  if (selection.resolution != viewport.region.size()) {
    m_retainedSelectionCode = SelectionBuffer::k_noSelectionCode;
    return false;
  }

  uint32_t entry = SelectionBuffer::k_noSelectionCode;

  // Each selection code has a priority, lower values means higher priority.
  // I pick region around the cursor, the size of which is a arbitrary.
  // Depending on the purpose of an app, the size of the region should be
  // dictated by accessibility guidelines. The purpose of the region is to
  // allow to select thin elements, without having to be precise. I wouldn’t
  // want to draw thick “Lego Duplo” like lines just to be able to select
  // them. I do like the visual elegance of thin lines. At the same time,
  // selection codes do need a priority, I cannot just pick the code in the
  // nearest non-empty pixel.

  // Render to texture
  GLint last_framebuffer;
  glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &last_framebuffer);
  check_gl("get last framebuffer");
  glBindFramebuffer(GL_FRAMEBUFFER, selection.frameBuffer);
  {
    // LOG_DEBUG << "Picking viewRegion " << viewRegion.lower.x << " " << viewRegion.lower.y << " " <<
    // viewRegion.upper.x
    //           << " " << viewRegion.upper.y;
    // LOG_DEBUG << "Picking region " << region.lower.x << " " << region.lower.y << " " << region.upper.x << " "
    //           << region.upper.y;
    glViewport(viewRegion.lower.x, viewRegion.lower.y, viewRegion.upper.x + 1, viewRegion.upper.y + 1);

    glm::ivec2 regionSize = region.size() + glm::ivec2(1);
    size_t size = size_t(regionSize.x) * size_t(regionSize.y);
    if (size) {
      // Read pixels over a region. What we read is an 32 bits unsigned partitioned into 8 bits
      // RGB values... at least until we figure out how to do it better.
      // If selection region is small, work on stack memory, otherwise allocate.
      uint8_t valuesLocalBuffer[1024 * 4];
      uint8_t* values = (size <= 1024 ? valuesLocalBuffer : (uint8_t*)malloc(size * 4));
      glReadPixels(region.lower.x, region.lower.y, regionSize.x, regionSize.y, GL_RGBA, GL_UNSIGNED_BYTE, values);
      check_gl("readpixels");

      // Search the click area for the lowest selection code. Lower code means
      // higher selection priority.
      for (uint8_t* rgba = values; rgba < (values + size * 4); rgba += 4) {
        uint32_t code = selectionRGB8ToCode(rgba);
        if (code != SelectionBuffer::k_noSelectionCode) {
          if (code < entry) {
            entry = code;
          }
        }
      }

      if (values != valuesLocalBuffer) {
        free(values);
      }
    }
  }
  // Restore previous framebuffer
  glBindFramebuffer(GL_FRAMEBUFFER, last_framebuffer);
  check_gl("restore framebuffer");

  // if (entry < SelectionBuffer::k_noSelectionCode) {
  //   LOG_DEBUG << "Selection: " << entry;
  // }
  m_retainedSelectionCode = entry;
  return entry != SelectionBuffer::k_noSelectionCode;
}

void
Gesture::drawArc(const glm::vec3& pstart,
                 float angle,
                 const glm::vec3& center,
                 const glm::vec3& normal,
                 uint32_t numSegments,
                 glm::vec3 color,
                 float opacity,
                 uint32_t code)
{
  // draw arc from pstart through angle with center of circle at center
  glm::vec3 xaxis = pstart - center;
  glm::vec3 yaxis = glm::cross(normal, xaxis);
  for (int i = 0; i < numSegments; ++i) {
    float t0 = float(i) / float(numSegments);
    float t1 = float(i + 1) / float(numSegments);

    float theta0 = t0 * angle; // 2.0f * glm::pi<float>();
    float theta1 = t1 * angle; // 2.0f * glm::pi<float>();

    glm::vec3 p0 = center + xaxis * cosf(theta0) + yaxis * sinf(theta0);
    glm::vec3 p1 = center + xaxis * cosf(theta1) + yaxis * sinf(theta1);

    graphics.addLine(Gesture::Graphics::VertsCode(p0, color, opacity, code),
                     Gesture::Graphics::VertsCode(p1, color, opacity, code));
  }
}

void
Gesture::drawCircle(glm::vec3 center,
                    glm::vec3 xaxis,
                    glm::vec3 yaxis,
                    uint32_t numSegments,
                    glm::vec3 color,
                    float opacity,
                    uint32_t code,
                    glm::vec4* clipPlane)
{
  for (int i = 0; i < numSegments; ++i) {
    float t0 = float(i) / float(numSegments);
    float t1 = float(i + 1) / float(numSegments);

    float theta0 = t0 * 2.0f * glm::pi<float>();
    float theta1 = t1 * 2.0f * glm::pi<float>();

    glm::vec3 p0 = center + xaxis * cosf(theta0) + yaxis * sinf(theta0);
    glm::vec3 p1 = center + xaxis * cosf(theta1) + yaxis * sinf(theta1);

    if (clipPlane) {
      if (glm::dot(*clipPlane, glm::vec4(p0, 1.0)) > 0 && glm::dot(*clipPlane, glm::vec4(p1, 1.0)) > 0) {
        graphics.addLine(Gesture::Graphics::VertsCode(p0, color, opacity, code),
                         Gesture::Graphics::VertsCode(p1, color, opacity, code));
      }
    } else {
      graphics.addLine(Gesture::Graphics::VertsCode(p0, color, opacity, code),
                       Gesture::Graphics::VertsCode(p1, color, opacity, code));
    }
  }
}

// does not draw a flat base
void
Gesture::drawCone(glm::vec3 base,
                  glm::vec3 xaxis,
                  glm::vec3 yaxis,
                  glm::vec3 zaxis,
                  uint32_t numSegments,
                  glm::vec3 color,
                  float opacity,
                  uint32_t code)
{
  for (int i = 0; i < numSegments; ++i) {
    float t0 = float(i) / float(numSegments);
    float t1 = float(i + 1) / float(numSegments);

    float theta0 = t0 * 2.0f * glm::pi<float>();
    float theta1 = t1 * 2.0f * glm::pi<float>();

    glm::vec3 p0 = base + xaxis * cosf(theta0) + yaxis * sinf(theta0);
    glm::vec3 p1 = base + xaxis * cosf(theta1) + yaxis * sinf(theta1);

    graphics.addVert(Gesture::Graphics::VertsCode(base + zaxis, color, opacity, code));
    graphics.addVert(Gesture::Graphics::VertsCode(p1, color, opacity, code));
    graphics.addVert(Gesture::Graphics::VertsCode(p0, color, opacity, code));
  }
}

void
Gesture::drawText(std::string stext, glm::vec3 p, glm::vec2 scale, glm::vec3 color, float opacity, uint32_t code)
{
  float xpos = p.x;
  float ypos = p.y;

  // Currently gesture.graphics only supports one global texture for all draw commands.
  // This is safe for now because the font texture is the only one needed.
  // In future, if e.g. tool buttons need texture images, then we have to
  // attach the texture id with the draw command.
  graphics.glTextureId = graphics.font->getTextureID();

  // assume orthographic projection with units = screen pixels, origin at top left
  // also assume we are in a "TRIANGLES" draw command.

  stbtt_aligned_quad q;
  const char* text = stext.c_str();
  while (*text) {
    if (graphics.font->getBakedQuad(*text, &xpos, &ypos, &q)) {
      // apply scaling to q.x0, q.y0, q.x1, q.y1 relative to start position p
      q.x0 = p.x + (q.x0 - p.x) * scale.x;
      q.y0 = p.y + (q.y0 - p.y) * scale.y;
      q.x1 = p.x + (q.x1 - p.x) * scale.x;
      q.y1 = p.y + (q.y1 - p.y) * scale.y;
      // QUAD.
      // 0
      graphics.addVert(
        Gesture::Graphics::VertsCode(glm::vec3(q.x0, q.y0, 0.0), glm::vec2(q.s0, q.t0), color, opacity, code));
      // 2
      graphics.addVert(
        Gesture::Graphics::VertsCode(glm::vec3(q.x1, q.y1, 0.0), glm::vec2(q.s1, q.t1), color, opacity, code));
      // 1
      graphics.addVert(
        Gesture::Graphics::VertsCode(glm::vec3(q.x1, q.y0, 0.0), glm::vec2(q.s1, q.t0), color, opacity, code));

      // 2
      graphics.addVert(
        Gesture::Graphics::VertsCode(glm::vec3(q.x1, q.y1, 0.0), glm::vec2(q.s1, q.t1), color, opacity, code));
      // 0
      graphics.addVert(
        Gesture::Graphics::VertsCode(glm::vec3(q.x0, q.y0, 0.0), glm::vec2(q.s0, q.t0), color, opacity, code));
      // 3
      graphics.addVert(
        Gesture::Graphics::VertsCode(glm::vec3(q.x0, q.y1, 0.0), glm::vec2(q.s0, q.t1), color, opacity, code));
    }
    ++text;
  }
}
