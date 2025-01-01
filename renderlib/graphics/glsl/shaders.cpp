#include "shaders.h"

#include "shaders/basicVolume_frag_gen.hpp"
#include "shaders/basicVolume_vert_gen.hpp"
#include "shaders/copy_frag_gen.hpp"
#include "shaders/copy_vert_gen.hpp"
#include "shaders/flat_frag_gen.hpp"
#include "shaders/flat_vert_gen.hpp"
#include "shaders/gui_frag_gen.hpp"
#include "shaders/gui_vert_gen.hpp"
#include "shaders/imageNoLut_frag_gen.hpp"
#include "shaders/imageNoLut_vert_gen.hpp"
#include "shaders/pathTraceVolume_frag_gen.hpp"
#include "shaders/pathTraceVolume_vert_gen.hpp"
#include "shaders/ptAccum_frag_gen.hpp"
#include "shaders/ptAccum_vert_gen.hpp"
#include "shaders/thickLines_frag_gen.hpp"
#include "shaders/thickLines_vert_gen.hpp"
#include "shaders/toneMap_frag_gen.hpp"
#include "shaders/toneMap_vert_gen.hpp"

#include "Logging.h"

#include <memory>

struct ShaderEntry
{
  std::string src;
  GLenum type;
};

static std::unordered_map<std::string, ShaderEntry> shader_src = {
  { "basicVolumeVert", { basicVolume_vert_src, GL_VERTEX_SHADER } },
  { "basicVolumeFrag", { basicVolume_frag_src, GL_FRAGMENT_SHADER } },
  { "copyVert", { copy_vert_src, GL_VERTEX_SHADER } },
  { "copyFrag", { copy_frag_src, GL_FRAGMENT_SHADER } },
  { "flatVert", { flat_vert_src, GL_VERTEX_SHADER } },
  { "flatFrag", { flat_frag_src, GL_FRAGMENT_SHADER } },
  { "guiVert", { gui_vert_src, GL_VERTEX_SHADER } },
  { "guiFrag", { gui_frag_src, GL_FRAGMENT_SHADER } },
  { "imageNoLutVert", { imageNoLut_vert_src, GL_VERTEX_SHADER } },
  { "imageNoLutFrag", { imageNoLut_frag_src, GL_FRAGMENT_SHADER } },
  { "pathTraceVolumeVert", { pathTraceVolume_vert_src, GL_VERTEX_SHADER } },
  { "pathTraceVolumeFrag", { pathTraceVolume_frag_src, GL_FRAGMENT_SHADER } },
  { "ptAccumVert", { ptAccum_vert_src, GL_VERTEX_SHADER } },
  { "ptAccumFrag", { ptAccum_frag_src, GL_FRAGMENT_SHADER } },
  { "thickLinesVert", { thickLines_vert_src, GL_VERTEX_SHADER } },
  { "thickLinesFrag", { thickLines_frag_src, GL_FRAGMENT_SHADER } },
  { "toneMapVert", { toneMap_vert_src, GL_VERTEX_SHADER } },
  { "toneMapFrag", { toneMap_frag_src, GL_FRAGMENT_SHADER } },
};

static std::unordered_map<std::string, GLShader*> shaders;

bool
ShaderArray::CompileShaders()
{
  for (auto& shaderEntry : shader_src) {
    std::unique_ptr<GLShader> s = std::make_unique<GLShader>(shaderEntry.second.type);
    if (!s->compileSourceCode(shaderEntry.second.src.c_str())) {
      LOG_ERROR << "Failed to compile shader " << shaderEntry.first;
      return false;
    }
  }
  return true;
}

GLShader*
ShaderArray::GetShader(const std::string& name)
{
  if (shaders.find(name) == shaders.end()) {
    if (shader_src.find(name) == shader_src.end()) {
      LOG_ERROR << "Shader " << name << " not found";
      return nullptr;
    }
    GLShader* s = new GLShader(shader_src[name].type);
    if (s->compileSourceCode(shader_src[name].src.c_str())) {
      shaders[name] = s;
      // LOG_DEBUG << "Compiled shader " << name;
      return s;
    } else {
      LOG_ERROR << "Failed to compile shader " << name;
      delete s;
      return nullptr;
    }
  } else {
    return shaders[name];
  }
}

void
ShaderArray::DestroyShaders()
{
  for (auto& shader : shaders) {
    delete shader.second;
  }
  shaders.clear();
}