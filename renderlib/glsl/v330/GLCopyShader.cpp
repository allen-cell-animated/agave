#include "glad/glad.h"
#include "GLCopyShader.h"

#include <glm.h>
#include <gl/Util.h>

#include <iostream>
#include <sstream>


GLCopyShader::GLCopyShader():
    QOpenGLShaderProgram(),
    m_vshader(),
    m_fshader()
{
    m_vshader = new QOpenGLShader(QOpenGLShader::Vertex);
	m_vshader->compileSourceCode
	(R"(
#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 uv;

out vec2 vUv;
      
void main()
{
  vUv = uv;
  gl_Position = vec4( position, 1.0 );
}
	)");

    if (!m_vshader->isCompiled())
    {
        std::cerr << "GLCopyShader: Failed to compile vertex shader\n" << m_vshader->log().toStdString() << std::endl;
    }

    m_fshader = new QOpenGLShader(QOpenGLShader::Fragment);
    m_fshader->compileSourceCode
    (R"(
#version 330 core

uniform sampler2D tTexture0;
in vec2 vUv;
out vec4 out_FragColor;

void main()
{
  out_FragColor = texture(tTexture0, vUv);
}
    )");

    if (!m_fshader->isCompiled())
    {
        std::cerr << "GLCopyShader: Failed to compile fragment shader\n" << m_fshader->log().toStdString() << std::endl;
    }

    addShader(m_vshader);
    addShader(m_fshader);
    link();

    if (!isLinked())
    {
        std::cerr << "GLCopyShader: Failed to link shader program\n" << log().toStdString() << std::endl;
    }
	
    m_texture = uniformLocation("tTexture0");
}

GLCopyShader::~GLCopyShader()
{
}


void
GLCopyShader::setShadingUniforms()
{
    glUniform1i(m_texture, 0);
}
