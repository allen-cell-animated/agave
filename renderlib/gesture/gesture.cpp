#include "gesture.h"
#include "gl/Util.h"

#include <QApplication> // for doubleClickInterval

// Verify loose coupling between Gesture and GLFW
// static_assert(Gesture::Input::kButtonLeft   == GLFW_MOUSE_BUTTON_LEFT  , "Verify pointer buttons mapping");
// static_assert(Gesture::Input::kButtonRight  == GLFW_MOUSE_BUTTON_RIGHT , "Verify pointer buttons mapping");
// static_assert(Gesture::Input::kButtonMiddle == GLFW_MOUSE_BUTTON_MIDDLE, "Verify pointer buttons mapping");

// static_assert(Gesture::Input::kShift == GLFW_MOD_SHIFT  , "Verify pointer modifiers mapping");
// static_assert(Gesture::Input::kCtrl  == GLFW_MOD_CONTROL, "Verify pointer modifiers mapping");
// static_assert(Gesture::Input::kAlt   == GLFW_MOD_ALT    , "Verify pointer modifiers mapping");
// static_assert(Gesture::Input::kSuper == GLFW_MOD_SUPER  , "Verify pointer modifiers mapping");

// platform-specific
static double
gestureGetDoubleClickTime()
{
  auto millisec = QApplication::doubleClickInterval();

  return double(millisec) / 1000.0;
}

// During app initialization query the OS accessibility settings how the user configured the
// double-click duration. Developer: never hardcode this time to something that feels
// right to you.
double Gesture::Input::s_doubleClickTime = gestureGetDoubleClickTime();

// Update the current action for one of the button of the pointer device
void
Gesture::Input::setButtonEvent(uint32_t mbIndex, Action action, int mods, glm::vec2 position, double time)
{
  if (mbIndex >= kButtonsCount)
    return;

  Button& button = mbs[mbIndex];
  if (action == Input::kPress) {
    // If the the button is pressed and was previously in a neutral state, it could be a click or a
    // double-click.
    if (button.action == Gesture::Input::kNone) {
      // A double-click event is recorded if the new click follows a previous click of the
      // same button within some time interval.
      button.doubleClick = (time - button.triggerTime) < Input::s_doubleClickTime;
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
    if (button.action == Input::kNone || button.action == Input::kRelease)
      continue;

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
configure_3dDepthTested(SceneView& sceneView)
{
  auto& shaders = sceneView.shaders;

  glm::mat4 v(1.0);
  sceneView.camera.getViewMatrix(v);
  glm::mat4 p(1.0);
  sceneView.camera.getProjMatrix(p);

  // glm::mat4 s = glm::scale(glm::mat4(1.0), glm::vec3(1.0, -1.0, 1.0));
  glUniformMatrix4fv(shaders->gui.m_loc_proj, 1, GL_FALSE, glm::value_ptr(p * v));
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
configure_3dStacked(SceneView& sceneView)
{
  auto& shaders = sceneView.shaders;

  glm::mat4 v(1.0);
  sceneView.camera.getViewMatrix(v);
  glm::mat4 p(1.0);
  sceneView.camera.getProjMatrix(p);
  check_gl("PRE set proj matrix");

  // glm::mat4 s = glm::scale(glm::mat4(1.0), glm::vec3(1.0, -1.0, 1.0));
  glUniformMatrix4fv(shaders->gui.m_loc_proj, 1, GL_FALSE, glm::value_ptr(p * v));

  check_gl("set proj matrix");

  glDisable(GL_DEPTH_TEST);
  check_gl("disable depth test");
}

// The third pass is a 2d orthographic projection of screen space, where the
// coordinates are measured in pixels starting at the lower left corner of the
// screen. Here is where I draw buttons or other traditional GUI elements if you wish.

// Draw something in screen space without zbuffer.
static void
configure_2dScreen(SceneView& sceneView)
{
  auto& shaders = sceneView.shaders;

  auto p = glm::ortho((float)sceneView.viewport.region.lower.x,
                      (float)sceneView.viewport.region.upper.x,
                      (float)sceneView.viewport.region.lower.y,
                      (float)sceneView.viewport.region.upper.y,
                      1.0f,
                      -1.f);
  glUniformMatrix4fv(shaders->gui.m_loc_proj, 1, GL_FALSE, glm::value_ptr(p));
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
    glClearColor(1, 1, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    drawSceneGeometry();
  }

  // Restore
  glBindFramebuffer(GL_FRAMEBUFFER, last_framebuffer);
  check_gl("restore default framebuffer");
  if (last_enable_depth_test)
    glEnable(GL_DEPTH_TEST);
  else
    glDisable(GL_DEPTH_TEST);
  check_gl("restore depth test state");
  if (last_enable_blend)
    glEnable(GL_BLEND);
  else
    glDisable(GL_BLEND);
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
Gesture::Graphics::draw(SceneView& sceneView, const SelectionBuffer& selection)
{
  // Gesture draw spans across the entire window and it is not restricted to a single
  // viewport.
  if (this->verts.empty()) {
    clearCommands();
    return;
  }

  // YAGNI: With a small effort we could create dynamic passes that are
  //        fully user configurable...
  //
  // Configure command lists
  void (*pipelineConfig[3])(SceneView&);
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
      auto& shaders = sceneView.shaders;
      shaders->gui.configure(display, this->glTextureId);

      for (int sequence = 0; sequence < Graphics::kNumCommandsLists; ++sequence) {
        if (!this->commands[sequence].empty()) {
          pipelineConfig[sequence](sceneView);

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

      shaders->gui.cleanup();
      check_gl("disablevertexattribarray");
    };
    drawGesture(/*display*/ true);

    // The last thing we draw is selection codes for next frame. This allows us
    // to know what is under the pointer cursor.
    drawGestureCodes(selection, sceneView.viewport, [&]() { drawGesture(/*display*/ false); });
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
  // Todo: the choice of pointer button should not be hardcoded here
  const Input::Button& button = input.mbs[Gesture::Input::kButtonLeft];

  int clickEnded = (button.action == Input::Action::kRelease);
  int clickDrag = (button.action == Input::Action::kDrag);
  int32_t buttonModifier = button.modifier;

  if (clickEnded || clickDrag) {
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
          if (code < entry)
            entry = code;
        }
      }

      if (values != valuesLocalBuffer)
        free(values);
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
