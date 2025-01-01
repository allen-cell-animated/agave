#include "glad/glad.h"

#include "GLPTAccumShader.h"
#include "shaders.h"

#include <gl/Util.h>
#include <glm.h>

#include <iostream>
#include <sstream>

GLPTAccumShader::GLPTAccumShader()
  : GLShaderProgram()
  , vshader()
  , fshader()
{
  vshader = ShaderArray::GetShader("ptAccumVert");

  if (!vshader->isCompiled()) {
    LOG_ERROR << "GLPTAccumShader: Failed to compile vertex shader\n" << vshader->log();
  }

  fshader = ShaderArray::GetShader("ptAccumFrag");

  if (!fshader->isCompiled()) {
    LOG_ERROR << "GLPTAccumShader: Failed to compile fragment shader\n" << fshader->log();
  }

  addShader(vshader);
  addShader(fshader);
  link();

  if (!isLinked()) {
    LOG_ERROR << "GLPTAccumShader: Failed to link shader program\n" << log();
  }

  uTextureRender = uniformLocation("textureRender");
  uTextureAccum = uniformLocation("textureAccum");

  uNumIterations = uniformLocation("numIterations");
}

GLPTAccumShader::~GLPTAccumShader() {}

void
GLPTAccumShader::setShadingUniforms()
{
  glUniform1i(uTextureRender, 0);
  glUniform1i(uTextureAccum, 1);
  glUniform1i(uNumIterations, numIterations);
}

void
GLPTAccumShader::setTransformUniforms(const CCamera& camera, const glm::mat4& modelMatrix)
{
}
