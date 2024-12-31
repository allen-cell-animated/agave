#include "GLCopyShader.h"
#include "glad/glad.h"

#include "Logging.h"
#include "shaders.h"

#include <gl/Util.h>
#include <glm.h>

#include <iostream>
#include <sstream>

GLCopyShader::GLCopyShader()
  : GLShaderProgram()
  , m_vshader()
  , m_fshader()
{
  m_vshader = ShaderArray::GetShader("copyVert");

  if (!m_vshader->isCompiled()) {
    LOG_ERROR << "GLCopyShader: Failed to compile vertex shader\n" << m_vshader->log();
  }

  m_fshader = ShaderArray::GetShader("copyFrag");

  if (!m_fshader->isCompiled()) {
    LOG_ERROR << "GLCopyShader: Failed to compile fragment shader\n" << m_fshader->log();
  }

  addShader(m_vshader);
  addShader(m_fshader);
  link();

  if (!isLinked()) {
    LOG_ERROR << "GLCopyShader: Failed to link shader program\n" << log();
  }

  m_texture = uniformLocation("tTexture0");
}

GLCopyShader::~GLCopyShader() {}

void
GLCopyShader::setShadingUniforms()
{
  glUniform1i(m_texture, 0);
}
