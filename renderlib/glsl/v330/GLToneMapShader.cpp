#include "glad/glad.h"
#include "GLToneMapShader.h"

#include <glm.h>
#include <gl/Util.h>

#include <iostream>
#include <sstream>


GLToneMapShader::GLToneMapShader():
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
        std::cerr << "GLToneMapShader: Failed to compile vertex shader\n" << m_vshader->log().toStdString() << std::endl;
    }

    m_fshader = new QOpenGLShader(QOpenGLShader::Fragment);
    m_fshader->compileSourceCode
    (R"(
#version 330 core

uniform float gInvExposure;
uniform sampler2D tTexture0;
in vec2 vUv;
out vec4 out_FragColor;
      
vec3 XYZtoRGB(vec3 xyz) {
    return vec3(
        3.240479f*xyz[0] - 1.537150f*xyz[1] - 0.498535f*xyz[2],
        -0.969256f*xyz[0] + 1.875991f*xyz[1] + 0.041556f*xyz[2],
        0.055648f*xyz[0] - 0.204043f*xyz[1] + 1.057311f*xyz[2]
    );
}
      
void main()
{
    vec4 pixelColor = texture(tTexture0, vUv);
    pixelColor.rgb = XYZtoRGB(pixelColor.rgb);
      
    pixelColor.rgb = 1.0-exp(-pixelColor.rgb*gInvExposure);
    pixelColor = clamp(pixelColor, 0.0, 1.0);
      
    out_FragColor = pixelColor;
    //out_FragColor = pow(pixelColor, vec4(1.0/2.2));
}
    )");

    if (!m_fshader->isCompiled())
    {
        std::cerr << "GLToneMapShader: Failed to compile fragment shader\n" << m_fshader->log().toStdString() << std::endl;
    }

    addShader(m_vshader);
    addShader(m_fshader);
    link();

    if (!isLinked())
    {
        std::cerr << "GLToneMapShader: Failed to link shader program\n" << log().toStdString() << std::endl;
    }
	
    m_texture = uniformLocation("tTexture0");
    m_InvExposure = uniformLocation("gInvExposure");
}

GLToneMapShader::~GLToneMapShader()
{
}


void
GLToneMapShader::setShadingUniforms(float invExposure)
{
  glUniform1i(m_texture, 0);
  glUniform1f(m_InvExposure, invExposure);
}
