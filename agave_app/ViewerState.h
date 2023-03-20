#pragma once

#include "Serialize.h"

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

Serialize::LoadSettings
fromLoadSpec(const LoadSpec& loadSpec);

Serialize::LightSettings_V1
fromLight(const Light& light);