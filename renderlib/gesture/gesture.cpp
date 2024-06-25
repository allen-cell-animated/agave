#include "gesture.h"
#include "gl/Util.h"
#include "graphics/GestureGraphicsGL.h"
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

void
Gesture::Graphics::draw(SceneView& sceneView, SelectionBuffer* selection)
{
  // Gesture draw spans across the entire window and it is not restricted to a single
  // viewport.
  if (this->verts.empty()) {
    clearCommands();

    // TODO: do this clear only once if verts empty on consecutive frames?
    // it would save some computation but this is really not a bottleneck here.
    if (selection) {
      selection->clear();
    }
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
