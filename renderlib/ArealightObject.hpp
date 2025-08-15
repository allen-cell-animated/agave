#pragma once

#include "core/prty/prtyObject.hpp"
#include "core/prty/prtyColor.hpp"
#include "core/prty/prtyFloat.hpp"
#include "uiInfo.hpp"

#include <memory>
#include <functional>

class SceneLight;

// Data object for arealight properties
class ArealightDataObject
{
public:
  ArealightDataObject() = default;

  prtyFloat Theta{ "Theta", 0.0f };
  prtyFloat Phi{ "Phi", 1.5708f }; // PI/2
  prtyFloat Size{ "Size", 1.0f };
  prtyFloat Distance{ "Distance", 10.0f };
  prtyFloat Intensity{ "Intensity", 100.0f };
  prtyColor Color{ "Color", glm::vec4(1.0f, 1.0f, 1.0f, 1.0f) };
};

class ArealightObject : public prtyObject
{
public:
  ArealightObject();

  // Set the scene light instance to control
  void setSceneLight(SceneLight* sceneLight) { m_sceneLight = sceneLight; }
  SceneLight* getSceneLight() const { return m_sceneLight; }

  // Set dirty callback for change notifications
  void setDirtyCallback(std::function<void()> callback) { m_dirtyCallback = callback; }

  // Update properties from scene light instance
  void updatePropsFromSceneLight();

  // Update scene light instance from properties (called automatically via callbacks)
  void updateSceneLightFromProps();

  // Property change callbacks
  void ThetaChanged(prtyProperty* i_Property, bool i_bDirty);
  void PhiChanged(prtyProperty* i_Property, bool i_bDirty);
  void SizeChanged(prtyProperty* i_Property, bool i_bDirty);
  void DistanceChanged(prtyProperty* i_Property, bool i_bDirty);
  void IntensityChanged(prtyProperty* i_Property, bool i_bDirty);
  void ColorChanged(prtyProperty* i_Property, bool i_bDirty);

private:
  ArealightDataObject m_arealightDataObject;
  SceneLight* m_sceneLight = nullptr;
  std::function<void()> m_dirtyCallback;

  // UI Info objects
  FloatSliderSpinnerUiInfo* m_thetaUIInfo = nullptr;
  FloatSliderSpinnerUiInfo* m_phiUIInfo = nullptr;
  FloatSliderSpinnerUiInfo* m_sizeUIInfo = nullptr;
  FloatSliderSpinnerUiInfo* m_distanceUIInfo = nullptr;
  FloatSliderSpinnerUiInfo* m_intensityUIInfo = nullptr;
  ColorPickerUiInfo* m_colorUIInfo = nullptr;
};
