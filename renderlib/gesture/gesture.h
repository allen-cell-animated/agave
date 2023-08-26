#pragma once
#include "glad/glad.h"

#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

#include "BoundingBox.h"
#include "CCamera.h"
#include "SceneView.h"
#include "graphics/gl/Util.h"
#include "graphics/glsl/GLGuiShader.h"

#include <memory>
#include <vector>

class GLGuiShader;
class Scene;
class RenderSettings;

// https://maxliani.wordpress.com/2021/06/06/offline-to-realtime-gesture/

struct Gesture
{
  // all mouse positions are in window coordinates
  // premultiplied by devicepixelratio in the gui layer
  // with 0,0 at TOP left.
  struct Input
  {
    // User Input
    static constexpr size_t kButtonsCount = 3;

    // During app initialization query the OS accessibility settings how the user configured the
    // double-click duration. Developer: never hardcode this time to something that feels
    // right to you.  The initial value is just a nonzero default that is meant to be overriddent.
    double doubleClickTime = 0.5; //< Mouse double-click time in seconds.
    void setDoubleClickTime(double seconds) { doubleClickTime = seconds; }

    enum ButtonId
    {
      kButtonLeft = 0,
      kButtonRight = 1,
      kButtonMiddle = 2
    };
    enum Mods
    {
      kShift = 0x0001,
      kCtrl = 0x0002,
      kAlt = 0x0004,
      kSuper = 0x0008,
      kCtrlShift = kCtrl | kShift
    };
    enum Action
    {
      kNone = 0,
      kPress = 1,  //< The pointer button had been pressed during this update cycle
      kDrag = 2,   //< The pointer changed its position from where a button was pressed
      kRelease = 3 //< The pointer button got released
    };
    enum DragConstrain
    {
      kUnconstrained = 0,
      kHorizontal = 1,
      kVertical = 2
    };

    // Call to update the action state of a pointer button.
    void setButtonEvent(uint32_t buttonIndex, Action action, int mods, glm::vec2 position, double time);

    // Call to provide a new screen position of the pointer.
    void setPointerPosition(glm::vec2 position);

    void reset(int mbIndex)
    {
      assert(mbIndex < kButtonsCount);
      if (mbIndex >= kButtonsCount) {
        return;
      }

      reset(mbs[mbIndex]);
    }

    void reset()
    {
      for (int mbIndex = 0; mbIndex < kButtonsCount; ++mbIndex) {
        reset(mbs[mbIndex]);
      }
    }

    // Call this function at the end of a frame before polling new events.
    void consume()
    {
      // Any button release event that we didn't explicitly use must be consumed, and the button
      // state must be reset.
      for (int mbIndex = 0; mbIndex < kButtonsCount; ++mbIndex) {
        if (mbs[mbIndex].action == kRelease) {
          reset(mbs[mbIndex]);
        }
      }
    }

    struct Button
    {
      Action action = kNone;
      // A bitwise combination of Mods keys (such as alt, ctrl, shift, etc...)
      int modifier = 0;
      // The screen coordinates the button was initially pressed, zero if no action
      glm::vec2 pressedPosition = {};
      // The drag vector from pressedPosition, this is still valid if pressed is kRelease and
      // it can be used to differentiate between a click action to be execute on button release
      // and the end of a click-drag gesture.
      glm::vec2 drag = {};
      // 0: unconstrained, 1: horizontal, 2: vertical
      DragConstrain dragConstraint = kUnconstrained;
      bool doubleClick = false;
      // The last time the button was pressed. This is used to detect double-click events.
      double triggerTime = 0;
    };

    Button mbs[kButtonsCount];
    glm::vec2 cursorPos = glm::vec2(0);

    bool hasButtonAction(ButtonId id, int mods)
    {
      return mbs[id].action != kNone && (mods == 0 || ((mbs[id].modifier & mods) > 0));
    }

    // Reset is typically executed by the command or tool that consumes the button release action.
    static void reset(Button& button)
    {
      button.action = kNone;
      button.modifier = 0;
      button.pressedPosition = glm::vec2(0);
      button.drag = glm::vec2(0);
      button.dragConstraint = kUnconstrained;
      button.doubleClick = false;
      // Do not reset triggerTime here: we need to retain its value to detect double-clicks
    }
  };
  Input input;

  // A minimalist graphics system to draw interactive GUI elements on screen
  struct Graphics
  {
    static constexpr int kInvalidVertexIndex = -1;

    // We support multiple command lists. Each list will draw with different
    // settings. The number of independent list is currently hardcoded but
    // it could be made configurable if we would store an array of function
    // pointers to setup each draw pass.
    enum class CommandSequence : int
    {
      k3dDepthTested = 0, //< for viewport depth compositing 3d GUI elements
      k3dStacked = 1,     //< for 3d GUI elements overlay
      k2dScreen = 2       //< for topmost 2d elements (widgets and buttons)
    };
    static constexpr int kNumCommandsLists = 3;

    struct Command
    {
      Command() = default;
      Command(GLenum command, float thickness = 1)
        : command(command)
        , thickness(thickness)
      {
      }

      GLenum command;  //< Any of GL_POINTS, GL_LINES, GL_TRIANGLES, etc...
      float thickness; //< Line thickness or point radius.
    };
    struct CommandRange
    {
      Command command;
      int begin, end;
    };

    // A struct to express a vertex as used to draw GUI geometry.
    struct VertsCode
    {
      // Some handshakes with the shader code about texture coords
      static constexpr float k_noTexture = -64.f;
      static constexpr float k_marqueePattern = -128.f;

      VertsCode() {}

      // Constructor commonly used for non textured elements
      VertsCode(const glm::vec3& v, glm::vec3 c, float opacity = 1.0f, uint32_t selectionCode = 0)
        : x(v.x)
        , y(v.y)
        , z(v.z)
        , u(k_noTexture)
        , v(k_noTexture)
        , r(c.x)
        , g(c.y)
        , b(c.z)
        , a(opacity)
        , s(selectionCode)
      {
      }

      // Comprehensive constructor
      VertsCode(const glm::vec3& v, glm::vec2 uv, glm::vec3 c, float opacity = 1.0f, uint32_t selectionCode = 0)
        : x(v.x)
        , y(v.y)
        , z(v.z)
        , u(uv.x)
        , v(uv.y)
        , r(c.x)
        , g(c.y)
        , b(c.z)
        , a(opacity)
        , s(selectionCode)
      {
      }

      float x, y, z;    //< position
      float u, v;       //< texture coordinates
      float r, g, b, a; //< draw color (multiplies texture)

      // A selection code used for UI interactions through the pointer cursor.
      // The selection code is drawn to a separate frame buffer for picking and it
      // is not affected by opacity.
      uint32_t s;

      // Note, this data layout is not optimal. We should go for a multiple
      // of 16 bytes, which could be achieved by replacing rgba  floating point
      // values with a single 32 bit int, plus 4 bytes of padding.
      // Without any such optimization the GUI seems to draw in a fraction of a
      // millisecond, so we are going to keep it simple for now.
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
        return create(resolution, samples);
      }
      void destroy();
      bool create(glm::ivec2 resolution, int samples = 0);
    };

    struct SelectionBuffer : RenderBuffer
    {
      // 1 bit is reserved for component flags.
      static constexpr uint32_t k_noSelectionCode = 0x7fffffffu;
    };

    // Gesture draw
    double timeDelta; //< A frame time for GUI interaction animation
    const CommandRange* currentCommand = nullptr;
    int lineLoopBegin = kInvalidVertexIndex;
    std::vector<VertsCode> verts;
    std::vector<CommandRange> commands[kNumCommandsLists];
    std::unique_ptr<GLGuiShader> shader;

    // A texture atlas for GUI elements
    // TODO: use bindless textures
    uint32_t glTextureId = 0;

    // remember selection code to reuse while dragging
    uint32_t m_retainedSelectionCode = SelectionBuffer::k_noSelectionCode;

    // Empty the commands/verts buffers, typically done after drawing the GUI.
    void clearCommands()
    {
      lineLoopBegin = kInvalidVertexIndex;
      currentCommand = nullptr;

      verts.clear();
      for (int i = 0; i < kNumCommandsLists; ++i) {
        commands[i].clear();
      }
    }

    // Add a draw command. There are multiple command sequences you can add to so
    // to handle 3d objects, 2d overlay, etc...
    // Commands are added *before* defining their geometry
    inline void addCommand(Command c, CommandSequence index = CommandSequence::k3dStacked)
    {
      // Terminate any previous command by recording the index of the last vertex
      // before starting a new command
      for (int i = 0; i < kNumCommandsLists; ++i) {
        if (!commands[i].empty()) {
          if (commands[i].back().end == kInvalidVertexIndex) {
            commands[i].back().end = verts.size();
          }
        }
      }

      int begin = verts.size();
      int end = kInvalidVertexIndex; //< leave the command buffer open to any new vertex

      commands[static_cast<int>(index)].push_back({ c, begin, end });
      currentCommand = &commands[static_cast<int>(index)].back();
    }

    // Any of GL_POINTS, GL_LINES, GL_TRIANGLES, etc...
    // TODO: abstract the command type into an api-independent enum
    inline void addCommand(GLenum command, CommandSequence index = CommandSequence::k3dStacked)
    {
      addCommand(Command(command), index);
    }

    // Add one vertex to the current command geometry
    inline void addVert(const VertsCode& v)
    {
      assert(currentCommand != nullptr);
      verts.push_back(v);
    }

    // Add two points to form a line, or the first line of a line loop.
    // To extend the line to form a polyline (open or close) use extLine.
    // It must be used with a GL_LINES command only.
    inline void addLine(const VertsCode& v0, const VertsCode& v1)
    {
      assert(currentCommand != nullptr);
      assert(currentCommand->command.command == GL_LINES);

      // Mark the spot for the first line vertex so that we can close a line
      // loop if we want to.
      lineLoopBegin = verts.size();
      verts.push_back(v0);
      verts.push_back(v1);
    }

    enum class LoopEntry : int
    {
      kContinue = 0,
      kClose = 1
    };
    // Create a polyline (open or close) by extending a line created with
    // addLine.
    inline void extLine(const VertsCode& v, LoopEntry loop = LoopEntry::kContinue)
    {
      assert(currentCommand != nullptr);
      assert(currentCommand->command.command == GL_LINES);

      // Since we draw lines, each line has two points. Repeat last vertex as the
      // first of this new line. We could make this more efficient by not repeating
      // points and draw with indices. But for the tests I have done so far this
      // seems to be plenty fast, so I'll keep this simple.
      verts.push_back(verts.back());
      verts.push_back(v);

      // Check if we need and can close a line loop.
      if (loop == LoopEntry::kClose && lineLoopBegin != kInvalidVertexIndex) {
        verts.push_back(v);
        verts.push_back(verts[lineLoopBegin]);
        lineLoopBegin = kInvalidVertexIndex;
      }
    }

    // Gesture draw, called once per window update (frame) when the GUI draw commands
    // had been described in full.
    void draw(struct SceneView& sceneView, const struct SelectionBuffer& selection);

    // Pick a GUI element using the cursor position in Input.
    // Return a valid GUI selection code, SelectionBuffer::k_noSelectionCode
    // otherwise.
    // viewport: left, top, width, height
    bool pick(struct SelectionBuffer& selection, const Input& input, const SceneView::Viewport& viewport);

    int getCurrentSelectionCode() { return m_retainedSelectionCode; }
  };
  Graphics graphics;
};
