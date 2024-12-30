#pragma once

#include "gl/Util.h"

class ShaderArray
{
public:
  static bool BuildShaders();
  static GLShader* GetShader(const std::string& name);
  static void DestroyShaders();
};
