#include "GLImageShader2DnoLut.h"

#include "Logging.h"

#include <gl/Util.h>
#include <glm.h>

#include <iostream>
#include <sstream>

GLImageShader2DnoLut::GLImageShader2DnoLut()
  : GLShaderProgram()
  , m_vshader()
  , m_fshader()
  , m_attr_coords()
  , m_attr_texcoords()
  , m_uniform_mvp()
  , m_uniform_texture()
{
  m_vshader = new GLShader(GL_VERTEX_SHADER);

  m_vshader->compileSourceCode("#version 400 core\n"
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

  if (!m_vshader->isCompiled()) {
    LOG_ERROR << "GLImageShader2DnoLut: Failed to compile vertex shader\n" << m_vshader->log();
  }

  m_fshader = new GLShader(GL_FRAGMENT_SHADER);
  m_fshader->compileSourceCode("#version 400 core\n"
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
                               "  vec4 texval = texture(tex, inData.f_texcoord);\n"
                               "\n"
                               "  outputColour = texval;\n"
                               "}\n");

  if (!m_fshader->isCompiled()) {
    LOG_ERROR << "GLImageShader2DnoLut: Failed to compile fragment shader\n" << m_fshader->log();
  }

  addShader(m_vshader);
  addShader(m_fshader);
  link();

  if (!isLinked()) {
    LOG_ERROR << "GLImageShader2DnoLut: Failed to link shader program\n" << log();
  }

  m_attr_coords = attributeLocation("coord2d");
  if (m_attr_coords == -1)
    LOG_ERROR << "GLImageShader2DnoLut: Failed to bind coordinates";

  m_attr_texcoords = attributeLocation("texcoord");
  if (m_attr_texcoords == -1)
    LOG_ERROR << "GLImageShader2DnoLut: Failed to bind texture coordinates";

  m_uniform_mvp = uniformLocation("mvp");
  if (m_uniform_mvp == -1)
    LOG_ERROR << "GLImageShader2DnoLut: Failed to bind transform";

  m_uniform_texture = uniformLocation("tex");
  if (m_uniform_texture == -1)
    LOG_ERROR << "GLImageShader2DnoLut: Failed to bind texture uniform ";
}

GLImageShader2DnoLut::~GLImageShader2DnoLut() {}

void
GLImageShader2DnoLut::enableCoords()
{
  enableAttributeArray(m_attr_coords);
  check_gl("enable attribute array: coords");
}

void
GLImageShader2DnoLut::disableCoords()
{
  disableAttributeArray(m_attr_coords);
  check_gl("disable attribute array: coords");
}

void
GLImageShader2DnoLut::setCoords(const GLfloat* offset, int tupleSize, int stride)
{
  setAttributeArray(m_attr_coords, offset, tupleSize, stride);
  check_gl("set attr coords pointer");
}

void
GLImageShader2DnoLut::setCoords(GLuint coords, const GLfloat* offset, int tupleSize, int stride)
{
  glBindBuffer(GL_ARRAY_BUFFER, coords);
  setCoords(offset, tupleSize, stride);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  check_gl("set attr coords buffer");
}

void
GLImageShader2DnoLut::enableTexCoords()
{
  enableAttributeArray(m_attr_texcoords);
  check_gl("enable attribute array texcoords");
}

void
GLImageShader2DnoLut::disableTexCoords()
{
  disableAttributeArray(m_attr_texcoords);
  check_gl("disable attribute array texcoords");
}

void
GLImageShader2DnoLut::setTexCoords(const GLfloat* offset, int tupleSize, int stride)
{
  setAttributeArray(m_attr_texcoords, offset, tupleSize, stride);
  check_gl("set attr texcoords ptr");
}

void
GLImageShader2DnoLut::setTexCoords(GLuint texcoords, const GLfloat* offset, int tupleSize, int stride)
{
  glBindBuffer(GL_ARRAY_BUFFER, texcoords);
  setTexCoords(offset, tupleSize, stride);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  check_gl("set attr texcoords buffer");
}

void
GLImageShader2DnoLut::setTexture(int texunit)
{
  glUniform1i(m_uniform_texture, texunit);
  check_gl("Set image texture");
}

void
GLImageShader2DnoLut::setModelViewProjection(const glm::mat4& mvp)
{
  glUniformMatrix4fv(m_uniform_mvp, 1, GL_FALSE, glm::value_ptr(mvp));
  check_gl("Set image2d uniform mvp");
}
