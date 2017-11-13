#include "glad/glad.h"
#include "V330GLFlatShader2D.h"
#include <glm.h>
#include <gl/Util.h>

#include <iostream>

GLFlatShader2D::GLFlatShader2D():
    QOpenGLShaderProgram(),
    vshader(),
    fshader(),
    attr_coords(),
    uniform_colour(),
    uniform_offset(),
    uniform_mvp()
{
    vshader = new QOpenGLShader(QOpenGLShader::Vertex);
    vshader->compileSourceCode
    ("#version 330 core\n"
        "\n"
        "uniform vec4 colour;\n"
        "uniform vec2 offset;\n"
        "uniform mat4 mvp;\n"
        "\n"
        "layout (location = 0) in vec2 coord2d;\n"
        "\n"
        "out VertexData\n"
        "{\n"
        "  vec4 f_colour;\n"
        "} outData;\n"
        "\n"
        "void main(void) {\n"
        "  gl_Position = mvp * vec4(coord2d+offset, 2.0, 1.0);\n"
        "  outData.f_colour = colour;\n"
        "}\n");
    if (!vshader->isCompiled())
    {
        std::cerr << "Failed to compile vertex shader\n" << vshader->log().toStdString() << std::endl;
    }

    fshader = new QOpenGLShader(QOpenGLShader::Fragment);
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
        std::cerr << "V330GLFlatShader2D: Failed to compile fragment shader\n" << fshader->log().toStdString() << std::endl;
    }

    addShader(vshader);
    addShader(fshader);
    link();

    if (!isLinked())
    {
        std::cerr << "V330GLFlatShader2D: Failed to link shader program\n" << log().toStdString() << std::endl;
    }

    attr_coords = attributeLocation("coord2d");
    if (attr_coords == -1)
    std::cerr << "V330GLFlatShader2D: Failed to bind coordinate location" << std::endl;

    uniform_colour = uniformLocation("colour");
    if (uniform_colour == -1)
    std::cerr << "V330GLFlatShader2D: Failed to bind colour" << std::endl;

    uniform_offset = uniformLocation("offset");
    if (uniform_offset == -1)
    std::cerr << "V330GLFlatShader2D: Failed to bind offset" << std::endl;

    uniform_mvp = uniformLocation("mvp");
    if (uniform_mvp == -1)
    std::cerr << "V330GLFlatShader2D: Failed to bind transform" << std::endl;
}

GLFlatShader2D::~GLFlatShader2D()
{
}

void
GLFlatShader2D::enableCoords()
{
    enableAttributeArray(attr_coords);
}

void
GLFlatShader2D::disableCoords()
{
    disableAttributeArray(attr_coords);
}

void
GLFlatShader2D::setCoords(const GLfloat *offset,
                            int            tupleSize,
                            int            stride)
{
    setAttributeArray(attr_coords, offset, tupleSize, stride);
    check_gl("Set flatcoords");
}

void
GLFlatShader2D::setCoords(GLuint  coords,
                            const GLfloat  *offset,
                            int             tupleSize,
                            int             stride)
{
	glBindBuffer(GL_ARRAY_BUFFER, coords);
    setCoords(offset, tupleSize, stride);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

void
GLFlatShader2D::setColour(const glm::vec4& colour)
{
    glUniform4fv(uniform_colour, 1, glm::value_ptr(colour));
    check_gl("Set flat uniform colour");
}

void
GLFlatShader2D::setOffset(const glm::vec2& offset)
{
    glUniform2fv(uniform_offset, 1, glm::value_ptr(offset));
    check_gl("Set flat uniform offset");
}

void
GLFlatShader2D::setModelViewProjection(const glm::mat4& mvp)
{
    glUniformMatrix4fv(uniform_mvp, 1, GL_FALSE, glm::value_ptr(mvp));
    check_gl("Set flat uniform mvp");
}

