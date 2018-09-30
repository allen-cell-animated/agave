#include "glad/glad.h"
#include "GLPTAccumShader.h"

#include <glm.h>
#include <gl/Util.h>

#include <iostream>
#include <sstream>


GLPTAccumShader::GLPTAccumShader():
    QOpenGLShaderProgram(),
    vshader(),
    fshader()
{
    vshader = new QOpenGLShader(QOpenGLShader::Vertex);
	vshader->compileSourceCode
	(R"(
#version 330 core

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

    if (!vshader->isCompiled())
    {
        std::cerr << "GLPTAccumShader: Failed to compile vertex shader\n" << vshader->log().toStdString() << std::endl;
    }

    fshader = new QOpenGLShader(QOpenGLShader::Fragment);
    fshader->compileSourceCode
    (R"(
#version 330 core

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

    if (!fshader->isCompiled())
    {
        std::cerr << "GLPTAccumShader: Failed to compile fragment shader\n" << fshader->log().toStdString() << std::endl;
    }

    addShader(vshader);
    addShader(fshader);
    link();

    if (!isLinked())
    {
        std::cerr << "GLPTAccumShader: Failed to link shader program\n" << log().toStdString() << std::endl;
    }
	

    uTextureRender = uniformLocation("textureRender");
    uTextureAccum = uniformLocation("textureAccum");

    uNumIterations = uniformLocation("numIterations");

}

GLPTAccumShader::~GLPTAccumShader()
{
}


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
	float w = (float)camera.m_Film.GetWidth();
	float h = (float)camera.m_Film.GetHeight();
	float vfov = camera.m_FovV * DEG_TO_RAD;

	glm::vec3 eye(camera.m_From.x, camera.m_From.y, camera.m_From.z);
	glm::vec3 center(camera.m_Target.x, camera.m_Target.y, camera.m_Target.z);
	glm::vec3 up(camera.m_Up.x, camera.m_Up.y, camera.m_Up.z);
	glm::mat4 cv = glm::lookAt(eye, center, up);
	glm::mat4 cp = glm::perspectiveFov(vfov, w, h, camera.m_Hither, camera.m_Yon);

	//glUniform3fv(uCameraPosition, 1, glm::value_ptr(camera.position));
	//glUniformMatrix4fv(uProjectionMatrix, 1, GL_FALSE, glm::value_ptr(cp));
	//glUniformMatrix4fv(uModelViewMatrix, 1, GL_FALSE, glm::value_ptr(cv * modelMatrix));
	//glUniformMatrix4fv(uInverseModelViewMatrix, 1, GL_FALSE, glm::value_ptr(glm::inverse(cv * modelMatrix)));
}
