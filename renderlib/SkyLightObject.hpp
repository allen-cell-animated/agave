#pragma once

#include "core/prty/prtyObject.hpp"
#include "core/prty/prtyColor.hpp"
#include "core/prty/prtyFloat.hpp"
#include "uiInfo.hpp"

#include <memory>
#include <functional>

class SceneLight;

// Data object for skylight properties
class SkylightDataObject
{
public:
  SkylightDataObject() = default;

  prtyFloat TopIntensity{ "TopIntensity", 1.0f };
  prtyColor TopColor{ "TopColor", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f) };
  prtyFloat MiddleIntensity{ "MiddleIntensity", 1.0f };
  prtyColor MiddleColor{ "MiddleColor", glm::vec4(0.5f, 0.5f, 0.5f, 1.0f) };
  prtyFloat BottomIntensity{ "BottomIntensity", 1.0f };
  prtyColor BottomColor{ "BottomColor", glm::vec4(0.2f, 0.2f, 0.2f, 1.0f) };
};

class SkyLightObject : public prtyObject
{
public:
  SkyLightObject();

  std::shared_ptr<SceneLight> getSceneLight() const { return m_sceneLight; }

  // Update properties from scene light instance
  void updatePropsFromSceneLight();

  // Update scene light instance from properties (called automatically via callbacks)
  void updateSceneLightFromProps();

  // Getter for data object
  SkylightDataObject& getDataObject() { return m_skylightDataObject; }
  const SkylightDataObject& getDataObject() const { return m_skylightDataObject; }

  // Property change callbacks
  void TopIntensityChanged(prtyProperty* i_Property, bool i_bDirty);
  void TopColorChanged(prtyProperty* i_Property, bool i_bDirty);
  void MiddleIntensityChanged(prtyProperty* i_Property, bool i_bDirty);
  void MiddleColorChanged(prtyProperty* i_Property, bool i_bDirty);
  void BottomIntensityChanged(prtyProperty* i_Property, bool i_bDirty);
  void BottomColorChanged(prtyProperty* i_Property, bool i_bDirty);

  void setDirtyCallback(std::function<void()> callback) { m_dirtyCallback = callback; }

  // document reading and writing; TODO consider an abstract base class to enforce commonality
  static constexpr uint32_t CURRENT_VERSION = 1;
  void fromDocument(docReader* reader);
  void toDocument(docWriter* writer);

private:
  SkylightDataObject m_skylightDataObject;

  // the actual object
  std::shared_ptr<SceneLight> m_sceneLight;

  std::function<void()> m_dirtyCallback;

  // UI Info objects
  ColorWithIntensityUiInfo* m_topIntensityUIInfo = nullptr;
  // ColorPickerUiInfo* m_topColorUIInfo = nullptr;
  ColorWithIntensityUiInfo* m_middleIntensityUIInfo = nullptr;
  // ColorPickerUiInfo* m_middleColorUIInfo = nullptr;
  ColorWithIntensityUiInfo* m_bottomIntensityUIInfo = nullptr;
  // ColorPickerUiInfo* m_bottomColorUIInfo = nullptr;
};
