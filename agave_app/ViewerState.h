#pragma once

#include "Serialize.h"
#include "renderDialog.h"

#include "renderlib/GradientData.h"
#include "renderlib/IFileReader.h"
#include "renderlib/json/json.hpp"

#include <glm.h>

#include <QString>

struct Light;

QString
stateToPythonScript(const Serialize::ViewerState& state);

Serialize::ViewerState
stateFromJson(const nlohmann::json& jsonDoc);

LoadSpec
stateToLoadSpec(const Serialize::ViewerState& state);

GradientData
stateToGradientData(const Serialize::ViewerState& state, int channelIndex);

Light
stateToLight(const Serialize::ViewerState& state, int lightIndex);

Serialize::LoadSettings
fromLoadSpec(const LoadSpec& loadSpec);

Serialize::LightSettings_V1
fromLight(const Light& light);

Serialize::CaptureSettings
fromCaptureSettings(const CaptureSettings& captureSettings);

Serialize::LutParams_V1
fromGradientData(const GradientData& lutParams);