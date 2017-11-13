#include "glad/glad.h"
#include "V330GLImageShader2DnoLut.h"

#include <glm.h>
#include <gl/Util.h>

#include <iostream>
#include <sstream>


GLImageShader2DnoLut::GLImageShader2DnoLut():
    QOpenGLShaderProgram(),
    vshader(),
    fshader(),
    attr_coords(),
    attr_texcoords(),
    uniform_mvp(),
    uniform_texture()
{
    vshader = new QOpenGLShader(QOpenGLShader::Vertex);

    vshader->compileSourceCode
    ("#version 330 core\n"
        "\n"
        "layout (location = 0) in vec2 coord2d;\n"
        "layout (location = 1) in vec2 texcoord;\n"
        "uniform mat4 mvp;\n"
        "\n"
        "out VertexData\n"
        "{\n"
        "  vec2 f_texcoord;\n"
        "} outData;\n"
        "\n"
        "void main(void) {\n"
        "  gl_Position = mvp * vec4(coord2d, 0.0, 1.0);\n"
        "  outData.f_texcoord = texcoord;\n"
        "}\n");

    if (!vshader->isCompiled())
    {
        std::cerr << "V330GLImageShader2DnoLut: Failed to compile vertex shader\n" << vshader->log().toStdString() << std::endl;
    }

    fshader = new QOpenGLShader(QOpenGLShader::Fragment);
    fshader->compileSourceCode
    ("#version 330 core\n"
        "\n"
        "uniform sampler2D tex;\n"
        "\n"
        "in VertexData\n"
        "{\n"
        "  vec2 f_texcoord;\n"
        "} inData;\n"
        "\n"
        "out vec4 outputColour;\n"
        "\n"
        "void main(void) {\n"
        "  vec2 flipped_texcoord = vec2(inData.f_texcoord.x, 1.0 - inData.f_texcoord.y);\n"
        "  vec4 texval = texture(tex, flipped_texcoord);\n"
        "\n"
        "  outputColour = texval;\n"
        "}\n");

    if (!fshader->isCompiled())
    {
        std::cerr << "V330GLImageShader2DnoLut: Failed to compile fragment shader\n" << fshader->log().toStdString() << std::endl;
    }

    addShader(vshader);
    addShader(fshader);
    link();

    if (!isLinked())
    {
        std::cerr << "V330GLImageShader2DnoLut: Failed to link shader program\n" << log().toStdString() << std::endl;
    }

    attr_coords = attributeLocation("coord2d");
    if (attr_coords == -1)
		std::cerr << "V330GLImageShader2DnoLut: Failed to bind coordinates" << std::endl;

    attr_texcoords = attributeLocation("texcoord");
    if (attr_texcoords == -1)
		std::cerr << "V330GLImageShader2DnoLut: Failed to bind texture coordinates" << std::endl;

    uniform_mvp = uniformLocation("mvp");
    if (uniform_mvp == -1)
		std::cerr << "V330GLImageShader2DnoLut: Failed to bind transform" << std::endl;

    uniform_texture = uniformLocation("tex");
    if (uniform_texture == -1)
		std::cerr << "V330GLImageShader2DnoLut: Failed to bind texture uniform " << std::endl;
}

GLImageShader2DnoLut::~GLImageShader2DnoLut()
{
}

void
GLImageShader2DnoLut::enableCoords()
{
    enableAttributeArray(attr_coords);
	check_gl("enable attribute array: coords");
}

void
GLImageShader2DnoLut::disableCoords()
{
    disableAttributeArray(attr_coords);
	check_gl("disable attribute array: coords");
}

void
GLImageShader2DnoLut::setCoords(const GLfloat *offset, int tupleSize, int stride)
{
    setAttributeArray(attr_coords, offset, tupleSize, stride);
	check_gl("set attr coords pointer");
}

void
GLImageShader2DnoLut::setCoords(GLuint coords, const GLfloat *offset, int tupleSize, int stride)
{
	glBindBuffer(GL_ARRAY_BUFFER, coords);
    setCoords(offset, tupleSize, stride);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	check_gl("set attr coords buffer");
}

void
GLImageShader2DnoLut::enableTexCoords()
{
    enableAttributeArray(attr_texcoords);
	check_gl("enable attribute array texcoords");
}

void
GLImageShader2DnoLut::disableTexCoords()
{
    disableAttributeArray(attr_texcoords);
	check_gl("disable attribute array texcoords");
}

void
GLImageShader2DnoLut::setTexCoords(const GLfloat *offset,
                                int            tupleSize,
                                int            stride)
{
    setAttributeArray(attr_texcoords, offset, tupleSize, stride);
	check_gl("set attr texcoords ptr");
}

void
GLImageShader2DnoLut::setTexCoords(GLuint  texcoords,
                                const GLfloat  *offset,
                                int             tupleSize,
                                int             stride)
{
	glBindBuffer(GL_ARRAY_BUFFER, texcoords);
    setTexCoords(offset, tupleSize, stride);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	check_gl("set attr texcoords buffer");
}

void
GLImageShader2DnoLut::setTexture(int texunit)
{
    glUniform1i(uniform_texture, texunit);
    check_gl("Set image texture");
}

void
GLImageShader2DnoLut::setModelViewProjection(const glm::mat4& mvp)
{
    glUniformMatrix4fv(uniform_mvp, 1, GL_FALSE, glm::value_ptr(mvp));
    check_gl("Set image2d uniform mvp");
}

