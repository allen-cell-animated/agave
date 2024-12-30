#include "shaders.h"

#include "shaders/basicVolume_frag_gen.hpp"

static std::unordered_map<std::string, std::string> shader_src = { { "shader", basicVolume_frag_shader } };
static std::unordered_map<std::string, GLShader*> shaders;

bool
ShaderArray::BuildShaders()
{
  for (auto& shader : shader_src) {
    GLShader* s = new GLShader(GL_FRAGMENT_SHADER);
    if (s->compileSourceCode(shader.second.c_str())) {
      shaders[shader.first] = s;
    } else {
      LOG_ERROR << "Failed to compile shader " << shader.first;
      return false;
    }
  }
  return true;
}

GLShader*
ShaderArray::GetShader(const std::string& name)
{
  return shaders[name];
}

void
ShaderArray::DestroyShaders()
{
  for (auto& shader : shaders) {
    delete shader.second;
  }
  shaders.clear();
}