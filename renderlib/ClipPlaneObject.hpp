#pragma once

#include "core/prty/prtyObject.hpp"
#include "core/prty/prtyBoolean.hpp"
#include "core/prty/prtyRotation.hpp"
#include "core/prty/prtyVector3d.hpp"
#include "uiInfo.hpp"

#include <memory>
#include <functional>

class ScenePlane;

// Data object for arealight properties
class ClipPlaneDataObject
{
public:
  ClipPlaneDataObject() = default;

  prtyBoolean Enabled{ "Enabled", true };
  prtyBoolean ShowHelper{ "ShowHelper", true };
  prtyVector3d Position{ "Position", glm::vec3(0.0f, 0.0f, 0.0f) };
  prtyRotation Rotation{ "Rotation", glm::vec3(0.0f, 0.0f, 0.0f) };
};

class ClipPlaneObject : public prtyObject
{
public:
  ClipPlaneObject();

  std::shared_ptr<ScenePlane> getScenePlane() const { return m_scenePlane; }

  // Update properties from scene plane instance
  void updatePropsFromObject();

  // Update scene plane instance from properties (called automatically via callbacks)
  void updateObjectFromProps();

  // Getter for data object
  ClipPlaneDataObject& getDataObject() { return m_clipPlaneDataObject; }
  const ClipPlaneDataObject& getDataObject() const { return m_clipPlaneDataObject; }

  // Property change callbacks
  void EnabledChanged(prtyProperty* i_Property, bool i_bDirty);
  void ShowHelperChanged(prtyProperty* i_Property, bool i_bDirty);
  void PositionChanged(prtyProperty* i_Property, bool i_bDirty);
  void RotationChanged(prtyProperty* i_Property, bool i_bDirty);

  void setDirtyCallback(std::function<void()> callback) { m_dirtyCallback = callback; }

  // document reading and writing; TODO consider an abstract base class to enforce commonality
  static constexpr uint32_t CURRENT_VERSION = 1;
  void fromDocument(docReader* reader);
  void toDocument(docWriter* writer);

private:
  ClipPlaneDataObject m_clipPlaneDataObject;

  // the actual object
  std::shared_ptr<ScenePlane> m_scenePlane;

  std::function<void()> m_dirtyCallback;

  // UI Info objects
  CheckBoxUiInfo* m_enabledUIInfo;
  CheckBoxUiInfo* m_showHelperUIInfo;
};
