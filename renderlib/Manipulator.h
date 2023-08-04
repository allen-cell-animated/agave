#pragma once

#include "MathUtil.h"
#include "gesture/gesture.h"

#include <assert.h>
#include <inttypes.h>
#include <vector>

struct ManipulationTool
{
  // Define the inactive selection code.
  static constexpr int kInactive = -1;

  ManipulationTool(const uint32_t numReservedCodes = 0)
    : m_activeCode(kInactive)
    , m_codesOffset(0)
    , m_numReservedCodes(numReservedCodes)
  {
  }

  /////////////////////////////////////////////////////////////////////////////////
  // Runtime methods executed per-frame by the system
  //
  // Several manipulators can coexist on screen on a particular frame, we call them
  // "active manipulators". Each of them requests a range of selection codes for
  // their own use.
  // Codes are in a continuous range and regenerated per frame. Absolute codes
  // are turned into tool-specific action codes for a manipulator to use.
  // Action codes begins at zero, kInactive (-1) is reserved for "no interaction".
  void requestCodesRange(int* incrementalCode)
  {
    assert(incrementalCode);
    m_codesOffset = *incrementalCode;
    *incrementalCode += m_numReservedCodes;
  }

  // User pointer interaction may activate a selection code; for example if the user
  // overs with the mouse on top of one of the manipulator features, the
  // corresponding code is broadcasted to all active manipulators. Each manipulator
  // is free to decide if the activeCode is meaningful and act on it, or if to
  // ignore the event. Typically this is done by checking if the activeCode against
  // the internal codes range.
  void setActiveCode(int code) { m_activeCode = std::max(kInactive, code - m_codesOffset); }

  // Once manipulators are executed (even if this results in no action), the
  // activeCode is reset to a neutral state. New codes will be broadcasted at the
  // beginning of each frame.
  void clear() { m_activeCode = kInactive; }

  /////////////////////////////////////////////////////////////////////////////////
  // Virtual interface, executed per-frame by the system
  //
  // Tools do something here. They read user interaction from the Gesture API, such
  //  as pointer click or drag information, and perform some action based on the
  // activeCode.
  virtual void action(SceneView& scene, Gesture& gesture) {}

  // After action execution (across all active manipulators). Tools can generate
  // draw geometry. If the implementation need to send some state across the two
  // calls, such as the value of an action just executed, it uses member variables
  // to retain the data for the duration of the frame. No actual draw happens here.
  // Draw command are accumulated in buffers and executed before end of frame.
  virtual void draw(SceneView& scene, Gesture& gesture) {}

  // During action execution, an active manipulator can display some message to the
  // app info line.
  static void displayInfoLine(const char*);

  static void destroyTool(ManipulationTool* tool) { delete tool; }
  // API End                                                                     //
  /////////////////////////////////////////////////////////////////////////////////
protected:
  // Test if activeCode is in range for the valid codes for this manipulator.
  bool isCodeValid(int code) const { return (code > kInactive) & (code < m_numReservedCodes); }

  // The action code selected by the user interaction, typically the one in
  // proximity of the cursor.
  int m_activeCode;

  // Manipulators action codes are relative codes, action codes + codesOffset give
  // absolute action codes for the system to process.
  int m_codesOffset;

  // Number of action codes reserved by an instance of a manipulator.
  // This is set on construction, but the implementation if free to change
  // the value per frame (typically inside method action).
  uint32_t m_numReservedCodes;

public:
  // Global manipulator scaling factor.
  // Todo: this shouldn't be here, instead a reference should be passed to
  //       those manipulators that needs such a setting: 3d manipulators or
  //       toolbars are likely to have different user configurations.
  static float s_manipulatorSize;

private:
  // Todo: add some API to describe the tool purpose, context and controls.
  //       We may want to use the information to extract tooltips or feed
  //       some App info line.
  std::string m_identifier;
};

struct Origins
{
  enum OriginFlags
  {
    kDefault = 0,
    kNormalize = 1
  };

  void clear() {}
  bool empty() const { return true; }
  void update(SceneView& scene) {}
  AffineSpace3f currentReference(SceneView& scene, OriginFlags flags = OriginFlags::kDefault) { return {}; }
};
struct MoveTool : ManipulationTool
{
  // Selection codes, are used to identify which manipulator is under the cursor.
  // The values in this enum are important, lower values means higher picking priority.
  enum MoveCodes
  {
    kMove = 0,
    kMoveX = 1,
    kMoveY = 2,
    kMoveZ = 3,
    kMoveYZ = 4,
    kMoveXZ = 5,
    kMoveXY = 6,
    kLast = 7
  };

  MoveTool()
    : ManipulationTool(kLast)
  {
  }

  virtual void action(SceneView& scene, Gesture& gesture) final;
  virtual void draw(SceneView& scene, Gesture& gesture) final;

  // Some data structure to store the initial state of the objects
  // to move.
  Origins origins;
};