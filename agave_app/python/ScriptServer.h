#pragma once

#include "command.h"

#include <string>

class ScriptServer
{
public:
  ScriptServer();
  ~ScriptServer();

  void runScriptFile(const std::string& path);
  void runScript(const std::string& source);

private:
};
