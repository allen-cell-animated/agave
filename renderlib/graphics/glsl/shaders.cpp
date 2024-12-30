#include "shaders.h"

#include "shaders/basicVolume_frag_gen.hpp"
#include "shaders/pathTraceVolume_frag_gen.hpp"

struct ShaderEntry
{
  std::string src;
  GLenum type;
};

static std::unordered_map<std::string, ShaderEntry> shader_src = {
  { "basicVolumeFrag", { basicVolume_frag_shader, GL_FRAGMENT_SHADER } },
  { "pathTraceVolumeFrag", { pathTraceVolume_frag_shader, GL_FRAGMENT_SHADER } }
};
static std::unordered_map<std::string, GLShader*> shaders;

bool
ShaderArray::BuildShaders()
{
  for (auto& shaderEntry : shader_src) {
    GLShader* s = new GLShader(shaderEntry.second.type);
    if (s->compileSourceCode(shaderEntry.second.src.c_str())) {
      shaders[shaderEntry.first] = s;
    } else {
      LOG_ERROR << "Failed to compile shader " << shaderEntry.first;
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