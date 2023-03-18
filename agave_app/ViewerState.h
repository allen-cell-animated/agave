#pragma once

#include "Serialize.h"

#include "renderlib/json/json.hpp"

#include <glm.h>

#include <QString>

QString
stateToPythonScript(const ViewerState&);

ViewerState
stateFromJson(const nlohmann::json& jsonDoc);
