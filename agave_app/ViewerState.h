#pragma once

#include "Serialize.h"

#include "renderlib/IFileReader.h"
#include "renderlib/json/json.hpp"


#include <glm.h>

#include <QString>

QString
stateToPythonScript(const Serialize::ViewerState& state);

Serialize::ViewerState
stateFromJson(const nlohmann::json& jsonDoc);

LoadSpec
stateToLoadSpec(const Serialize::ViewerState& state);
