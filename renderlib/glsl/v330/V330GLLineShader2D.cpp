/*
 * #%L
 * OME-QTWIDGETS C++ library for display of OME-Files pixel data and metadata.
 * %%
 * Copyright © 2014 - 2015 Open Microscopy Environment:
 *   - Massachusetts Institute of Technology
 *   - National Institutes of Health
 *   - University of Dundee
 *   - Board of Regents of the University of Wisconsin-Madison
 *   - Glencoe Software, Inc.
 * %%
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are
 * those of the authors and should not be interpreted as representing official
 * policies, either expressed or implied, of any organization.
 * #L%
 */

#include "V330GLLineShader2D.h"

#include <glm.h>
#include <gl/Util.h>

#include <iostream>

GLLineShader2D::GLLineShader2D():
    OpenGLShaderProgram(),
    vshader(),
    fshader(),
    attr_coords(),
    attr_colour(),
    uniform_mvp()
{
    vshader = new OpenGLShader(OpenGLShader::Vertex);
    vshader->compileSourceCode
    ("#version 330 core\n"
        "\n"
        "uniform mat4 mvp;\n"
        "uniform float zoom;\n"
        "\n"
        "layout (location = 0) in vec3 coord2d;\n"
        "layout (location = 1) in vec3 colour;\n"
        "out VertexData\n"
        "{\n"
        "  vec4 f_colour;\n"
        "} outData;\n"
        "\n"
        "void log10(in float v1, out float v2) { v2 = log2(v1) * 0.30103; }\n"
        "\n"
        "void main(void) {\n"
        "  gl_Position = mvp * vec4(coord2d[0], coord2d[1], -2.0, 1.0);\n"
        "  // Logistic function offset by LOD and correction factor to set the transition points\n"
        "  float logzoom;\n"
        "  log10(zoom, logzoom);\n"
        "  outData.f_colour = vec4(colour, 1.0 / (1.0 + pow(10.0,((-logzoom-1.0+coord2d[2])*30.0))));\n"
        "}\n");
    if (!vshader->isCompiled())
    {
        std::cerr << "V330GLLineShader2D: Failed to compile vertex shader\n" << vshader->log() << std::endl;
    }

    fshader = new OpenGLShader(OpenGLShader::Fragment);
    fshader->compileSourceCode
    ("#version 330 core\n"
        "\n"
        "in VertexData\n"
        "{\n"
        "  vec4 f_colour;\n"
        "} inData;\n"
        "\n"
        "out vec4 outputColour;\n"
        "\n"
        "void main(void) {\n"
        "  outputColour = inData.f_colour;\n"
        "}\n");
    if (!fshader->isCompiled())
    {
        std::cerr << "V330GLLineShader2D: Failed to compile fragment shader\n" << fshader->log() << std::endl;
    }

    addShader(vshader);
    addShader(fshader);
    link();

    if (!isLinked())
    {
        std::cerr << "V330GLLineShader2D: Failed to link shader program\n" << log() << std::endl;
    }

    attr_coords = attributeLocation("coord2d");
    if (attr_coords == -1)
    std::cerr << "V330GLLineShader2D: Failed to bind coordinate location" << std::endl;

    attr_colour = attributeLocation("colour");
    if (attr_coords == -1)
    std::cerr << "V330GLLineShader2D: Failed to bind colour location" << std::endl;

    uniform_mvp = uniformLocation("mvp");
    if (uniform_mvp == -1)
    std::cerr << "V330GLLineShader2D: Failed to bind transform" << std::endl;

    uniform_zoom = uniformLocation("zoom");
    if (uniform_zoom == -1)
    std::cerr << "V330GLLineShader2D: Failed to bind zoom factor" << std::endl;
}

GLLineShader2D::~GLLineShader2D()
{
}

void
GLLineShader2D::enableCoords()
{
    enableAttributeArray(attr_coords);
}

void
GLLineShader2D::disableCoords()
{
    disableAttributeArray(attr_coords);
}

void
GLLineShader2D::setCoords(const GLfloat *offset,
                            int            tupleSize,
                            int            stride)
{
    setAttributeArray(attr_coords, offset, tupleSize, stride);
}

void
GLLineShader2D::setCoords(GLuint coords,
                            const GLfloat *offset,
                            int            tupleSize,
                            int            stride)
{
    glBindBuffer(GL_ARRAY_BUFFER, coords);
    setCoords(offset, tupleSize, stride);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void
GLLineShader2D::enableColour()
{
    enableAttributeArray(attr_colour);
}

void
GLLineShader2D::disableColour()
{
    disableAttributeArray(attr_colour);
}

void
GLLineShader2D::setColour(const GLfloat *offset,
                            int            tupleSize,
                            int            stride)
{
    setAttributeArray(attr_colour, offset, tupleSize, stride);
}

void
GLLineShader2D::setColour(GLuint  colour,
                            const GLfloat  *offset,
                            int             tupleSize,
                            int             stride)
{
    glBindBuffer(GL_ARRAY_BUFFER, colour);
    setColour(offset, tupleSize, stride);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void
GLLineShader2D::setModelViewProjection(const glm::mat4& mvp)
{
    glUniformMatrix4fv(uniform_mvp, 1, GL_FALSE, glm::value_ptr(mvp));
    check_gl("Set line uniform mvp");
}

void
GLLineShader2D::setZoom(float zoom)
{
    glUniform1f(uniform_zoom, zoom);
    check_gl("Set line zoom level");
}
