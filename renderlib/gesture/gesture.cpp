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
GetDoubleClickTime()
{
  auto millisec = QApplication::doubleClickInterval();

  return double(millisec) / 1000.0;
}

// During app initialization query the OS accessibility settings how the user configured the
// double-click duration. Developer: never hardcode this time to something that feels
// right to you.
double Gesture::Input::s_doubleClickTime = GetDoubleClickTime();

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

const char* vertex_shader_text =
  R"(
    #version 150
    uniform mat4 projection;
    in vec3 vPos;
    in vec2 vUV;
    in vec4 vCol;
    out vec4 Frag_color;
    out vec2 Frag_UV;
 
    void main()
    {
        Frag_UV = vUV;
        Frag_color = vCol;
        gl_Position = projection * vec4(vPos, 1.0);
    }
    )";

const char* fragment_shader_text =
  R"(
    #version 150
    in vec4 Frag_color;
    in vec2 Frag_UV;
    in vec4 gl_FragCoord;
    uniform int picking;  //< draw for display or for picking? Picking has no texture.
    uniform sampler2D Texture;
    out vec4 outputF;
 
    void main()
    {
        vec4 result = Frag_color;
 
        // When drawing selection codes, everything is opaque.
        if (picking == 1)
            result.w = 1.0;
 
        // Gesture geometry handshake: any uv value below -64 means
        // no texture lookup. Check VertsCode::k_noTexture
        if (picking == 0 && Frag_UV.s > -64)
            result *= texture2D(Texture, Frag_UV.st);
 
        // Gesture geometry handshake: any uv equal to -128 means
        // overlay a checkerboard pattern. Check VertsCode::k_marqueePattern
        if (Frag_UV.s == -128)
        {
            // Create a pixel checkerboard pattern used for marquee
            // selection
            int x = int(gl_FragCoord.x); int y = int(gl_FragCoord.y);
            if (((x+y) & 1) == 0) result = vec4(0,0,0,1);
        }
        outputF = result;
    }
    )";

namespace Pipeline {
// Draw something "in the scene". This has a limitation that we assume there is a
// single viewport.
static void
configure_3dDepthTested(SceneView& sceneView)
{
  Shaders& shaders = sceneView.shaders;

  glUniformMatrix4fv(shaders.gui.loc_proj, 1, GL_FALSE, (const GLfloat*)sceneView.camera.xform.m);
  check_gl("set proj matrix");

  glEnable(GL_DEPTH_TEST);
  check_gl("enable depth test");
}
// Overlay something "in the scene". This has a limitation that we assume there
// is a single viewport.
static void
configure_3dStacked(SceneView& sceneView)
{
  Shaders& shaders = sceneView.shaders;

  glUniformMatrix4fv(shaders.gui.loc_proj, 1, GL_FALSE, (const GLfloat*)sceneView.camera.xform.m);
  check_gl("set proj matrix");

  glDisable(GL_DEPTH_TEST);
  check_gl("disable depth test");
}
// Draw something in screen space without zbuffer.
static void
configure_2dScreen(SceneView& sceneView)
{
  Shaders& shaders = sceneView.shaders;

  HomogeneousSpace4f p = HomogeneousSpace4f::ortho(sceneView.viewport.region.lower.x,
                                                   sceneView.viewport.region.upper.x,
                                                   sceneView.viewport.region.lower.y,
                                                   sceneView.viewport.region.upper.y,
                                                   1.0f,
                                                   -1.f);
  glUniformMatrix4fv(shaders.gui.loc_proj, 1, GL_FALSE, (const float*)p.m);
  check_gl("set proj matrix");

  glDisable(GL_DEPTH_TEST);
  check_gl("disable depth test");
}
} // namespace Pipeline

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

  // Draw UI and viewport manipulators
  {
    ScopedGlVertexBuffer vertex_buffer(this->verts.data(), this->verts.size() * sizeof(VertsCode));

    // Prepare a lambda to draw the Gesture commands. We'll run the lambda twice, once to
    // draw the GUI and once to draw the selection buffer data.
    auto drawGesture = [&](bool display) {
      Shaders& shaders = sceneView.shaders;
      shaders.gui.configure(display, this->glTextureId);

      for (int sequence = 0; sequence < Graphics::kNumCommandsLists; ++sequence) {
        if (!this->commands[sequence].empty()) {
          pipelineConfig[sequence](sceneView);

          // YAGNI: Commands could be coalesced, setting state could be avoided
          //        if not changing... For now it seems we can draw at over 2000 Hz
          //        and no further optimization is required.
          for (Graphics::CommandRange cmdr : this->commands[sequence]) {
            Graphics::Command& cmd = cmdr.cmd;
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

      shaders.gui.cleanup();
      glDisableVertexAttribArray(shaders.gui.loc_vpos);
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
  glToggle(GL_DEPTH_TEST, depthTest);
  check_gl("toggle depth test");

  clearCommands();
}

template<typename DrawBlock>
void
drawGestureCodes(const SelectionBuffer& selection, const glm::vec4& viewport, DrawBlock drawSceneGeometry)
{
  // Backup
  GLenum last_framebuffer;
  glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, (GLint*)&last_framebuffer);
  check_gl("get draw framebuffer");
  GLboolean last_enable_depth_test = glIsEnabled(GL_DEPTH_TEST);
  check_gl("is depth enabled");
  GLboolean last_enable_blend = glIsEnabled(GL_BLEND);
  check_gl("is blend enabled");

  // Render to texture
  glBindFramebuffer(GL_FRAMEBUFFER, selection.frameBuffer);
  check_gl("bind selection framebuffer");
  {
    glViewport(viewport.x, viewport.y, viewport.z, viewport.w);
    glEnable(GL_BLEND);
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
}

void
Gesture::RenderBuffer::destroy()
{
  if (frameBuffer == 0)
    return;
  glDeleteFramebuffers(1, &frameBuffer);
  glDeleteRenderbuffers(1, &depthRenderBuffer);
  glDeleteTextures(1, &renderedTexture);
  glDeleteTextures(1, &depthTexture);
  frameBuffer = 0;
  depthRenderBuffer = 0;
  renderedTexture = 0;
  resolution = Vec2i(0);
}

bool
Gesture::RenderBuffer::create(Vec2i resolution, int samples)
{
  this->resolution = resolution;
  this->samples = samples;

  glGenFramebuffers(1, &frameBuffer);
  glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);

  if (samples == 0) {
    glCreateTextures(GL_TEXTURE_2D, 1, &renderedTexture);

    // "Bind" the newly created texture: all future texture functions will modify this texture
    glBindTexture(GL_TEXTURE_2D, renderedTexture);

    // Define the texture quality and zeroes its memory
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, resolution.x, resolution.y, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);

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
    glCreateTextures(GL_TEXTURE_2D_MULTISAMPLE, 1, &renderedTexture);
    glCreateTextures(GL_TEXTURE_2D_MULTISAMPLE, 1, &depthTexture);

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

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  return status;
}

uint32_t
Gesture::Graphics::pick(SelectionBuffer& selection, const Gesture::Input& input, const glm::vec4& viewport)
{
  // Todo: the choice of pointer button should not be hardcoded here
  const Input::Button& button = input.mbs[Gesture::Input::kButtonLeft];
  int clikEnded = (button.action == Input::Action::kRelease);
  int clickDrag = (button.action == Input::Action::kDrag);
  int32_t buttonModifier = button.modifier;

  if (clikEnded || clickDrag)
    return SelectionBuffer::k_noSelectionCode; //< not a selection event

  // Prepare a region in raster spacer
  BBox2i region(wb::empty);
  {
    Vec2i pixel = viewport.toRaster(input.cursorPos);

    // Grow the click position by some pixels to improve usability. Ideally
    // this should be a configurable parameter to improve accessibility.
    constexpr int kClickRadius = 7; //< in pixels
    region.extend(pixel - Vec2i(kClickRadius));
    region.extend(pixel + Vec2i(kClickRadius));
  }

  // Render on the whole framebuffer, complete from the lower left corner tothe upper right
  BBox2i viewRegion(viewport.region.lower, viewport.region.upper - 1);

  // Crop selection with view in order to feed GL draw a valid region.
  region = intersect(region, viewRegion);

  // Frame buffer resolution should be correct, check just in case.
  if (selection.resolution != viewport.resolution)
    return SelectionBuffer::k_noSelectionCode;

  uint32_t entry = SelectionBuffer::k_noSelectionCode;

  // Render to texture
  GLint last_framebuffer;
  glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &last_framebuffer);
  check_gl("get last framebuffer");
  glBindFramebuffer(GL_FRAMEBUFFER, selection.frameBuffer);
  {
    glViewport(viewRegion.lower.x, viewRegion.lower.y, viewRegion.upper.x + 1, viewRegion.upper.y + 1);

    Vec2i regionSize = region.size() + Vec2i(1);
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
  check_gl("resture framebuffer");

  return entry;
}
