#include "GLCopyShader.h"
#include "glad/glad.h"

#include "Logging.h"

#include <gl/Util.h>
#include <glm.h>

#include <iostream>
#include <sstream>

GLCopyShader::GLCopyShader()
  : GLShaderProgram()
  , m_vshader()
  , m_fshader()
{
  m_vshader = new GLShader(GL_VERTEX_SHADER);
  m_vshader->compileSourceCode(R"(
#version 400 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 uv;

out vec2 vUv;
      
void main()
{
  vUv = uv;
  gl_Position = vec4( position, 1.0 );
}
	)");

  if (!m_vshader->isCompiled()) {
    LOG_ERROR << "GLCopyShader: Failed to compile vertex shader\n" << m_vshader->log();
  }

  m_fshader = new GLShader(GL_FRAGMENT_SHADER);
  m_fshader->compileSourceCode(R"(
#version 400 core

uniform sampler2D tTexture0;
in vec2 vUv;
out vec4 out_FragColor;

void main()
{
  out_FragColor = texture(tTexture0, vUv);
}
    )");

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
