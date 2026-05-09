#pragma once

#include "Serialize.h"
#include "renderDialog.h"

#include "renderlib/Colormap.h"
#include "renderlib/GradientData.h"
#include "renderlib/IFileReader.h"
#include "renderlib/json/json.hpp"

#include <glm.h>

#include <QString>

class Light;
class SceneLight;

QString
stateToPythonScript(const Serialize::ViewerState& state);

Serialize::ViewerState
stateFromJson(const nlohmann::json& jsonDoc);

LoadSpec
stateToLoadSpec(const Serialize::ViewerState& state);

GradientData
stateToGradientData(const Serialize::ViewerState& state, int channelIndex);

ColorRamp
stateToColorRamp(const Serialize::ViewerState& state, int channelIndex);

// Apply the serialized light settings at lightIndex to the given SceneLight,
// including the saved rotation quaternion. The SceneLight must already wrap a
// Light of the matching type, and its bounding-box-derived target must be set
// (e.g. via Scene::initBounds) before calling so the transform is centered
// correctly.
void
stateToLight(const Serialize::ViewerState& state, int lightIndex, SceneLight& sceneLight);

Serialize::LoadSettings
fromLoadSpec(const LoadSpec& loadSpec);

// Capture both the Light parameters and the SceneLight's rotation quaternion
// into a serializable LightSettings_V1.
Serialize::LightSettings_V1
fromLight(const SceneLight& sceneLight);

Serialize::CaptureSettings
fromCaptureSettings(const CaptureSettings& captureSettings, int viewWidth, int viewHeight);

Serialize::LutParams_V1
fromGradientData(const GradientData& lutParams);

Serialize::ColorMap
fromColorRamp(const ColorRamp& colorRamp);