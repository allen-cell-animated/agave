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

  // Set the scene light instance to control
  void setSceneLight(SceneLight* sceneLight) { m_sceneLight = sceneLight; }
  SceneLight* getSceneLight() const { return m_sceneLight; }

  // Set optional dirty callback for when properties change
  void setDirtyCallback(std::function<void()> callback) { m_dirtyCallback = callback; }

  // Update properties from scene light instance
  void updatePropsFromSceneLight();

  // Update scene light instance from properties (called automatically via callbacks)
  void updateSceneLightFromProps();

  // Property change callbacks
  void TopIntensityChanged(prtyProperty* i_Property, bool i_bDirty);
  void TopColorChanged(prtyProperty* i_Property, bool i_bDirty);
  void MiddleIntensityChanged(prtyProperty* i_Property, bool i_bDirty);
  void MiddleColorChanged(prtyProperty* i_Property, bool i_bDirty);
  void BottomIntensityChanged(prtyProperty* i_Property, bool i_bDirty);
  void BottomColorChanged(prtyProperty* i_Property, bool i_bDirty);

private:
  SkylightDataObject m_skylightDataObject;
  SceneLight* m_sceneLight = nullptr;
  std::function<void()> m_dirtyCallback;

  // UI Info objects
  FloatSliderSpinnerUiInfo* m_topIntensityUIInfo = nullptr;
  ColorPickerUiInfo* m_topColorUIInfo = nullptr;
  FloatSliderSpinnerUiInfo* m_middleIntensityUIInfo = nullptr;
  ColorPickerUiInfo* m_middleColorUIInfo = nullptr;
  FloatSliderSpinnerUiInfo* m_bottomIntensityUIInfo = nullptr;
  ColorPickerUiInfo* m_bottomColorUIInfo = nullptr;
};
