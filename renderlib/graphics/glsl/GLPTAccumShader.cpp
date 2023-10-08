#include "glad/glad.h"

#include "GLPTAccumShader.h"

#include <gl/Util.h>
#include <glm.h>

#include <iostream>
#include <sstream>

GLPTAccumShader::GLPTAccumShader()
  : GLShaderProgram()
  , vshader()
  , fshader()
{
  vshader = new GLShader(GL_VERTEX_SHADER);
  vshader->compileSourceCode(R"(
#version 400 core

layout (location = 0) in vec2 position;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
out VertexData
{
  vec2 pObj;
} outData;

void main()
{
  outData.pObj = position;
  gl_Position = vec4(position, 0.0, 1.0);
}
	)");

  if (!vshader->isCompiled()) {
    LOG_ERROR << "GLPTAccumShader: Failed to compile vertex shader\n" << vshader->log();
  }

  fshader = new GLShader(GL_FRAGMENT_SHADER);
  fshader->compileSourceCode(R"(
#version 400 core

in VertexData
{
  vec2 pObj;
} inData;

out vec4 outputColour;

uniform mat4 inverseModelViewMatrix;

uniform sampler2D textureRender;
uniform sampler2D textureAccum;

uniform int numIterations;

vec4 CumulativeMovingAverage(vec4 A, vec4 Ax, int N)
{
	 return A + ((Ax - A) / max(float(N), 1.0f));
}

void main()
{
    vec4 accum = textureLod(textureAccum, (inData.pObj+1.0) / 2.0, 0).rgba;
    vec4 render = textureLod(textureRender, (inData.pObj + 1.0) / 2.0, 0).rgba;

	//outputColour = accum + render;
	outputColour = CumulativeMovingAverage(accum, render, numIterations);
	return;
}
    )");

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
