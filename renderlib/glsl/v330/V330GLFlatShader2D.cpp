#include "V330GLFlatShader2D.h"
#include <gl/Util.h>
#include <glm.h>

#include "Logging.h"

#include <iostream>

GLFlatShader2D::GLFlatShader2D()
  : GLShaderProgram()
  , m_vshader()
  , m_fshader()
  , m_attr_coords()
  , m_uniform_colour()
  , m_uniform_mvp()
{
  m_vshader = new GLShader(GL_VERTEX_SHADER);
  m_vshader->compileSourceCode("#version 330 core\n"
                               "\n"
                               "uniform vec4 colour;\n"
                               "uniform mat4 mvp;\n"
                               "\n"
                               "layout (location = 0) in vec3 position;\n"
                               "\n"
                               "out VertexData\n"
                               "{\n"
                               "  vec4 f_colour;\n"
                               "} outData;\n"
                               "\n"
                               "void main(void) {\n"
                               "  gl_Position = mvp * vec4(position, 1.0);\n"
                               "  outData.f_colour = colour;\n"
                               "}\n");
  if (!m_vshader->isCompiled()) {
    LOG_ERROR << "Failed to compile vertex shader\n" << m_vshader->log();
  }

  m_fshader = new GLShader(GL_FRAGMENT_SHADER);
  m_fshader->compileSourceCode("#version 330 core\n"
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
  if (!m_fshader->isCompiled()) {
    LOG_ERROR << "V330GLFlatShader2D: Failed to compile fragment shader\n" << m_fshader->log();
  }

  addShader(m_vshader);
  addShader(m_fshader);
  link();

  if (!isLinked()) {
    LOG_ERROR << "V330GLFlatShader2D: Failed to link shader program\n" << log();
  }

  m_attr_coords = attributeLocation("coord2d");
  if (m_attr_coords == -1)
    LOG_ERROR << "V330GLFlatShader2D: Failed to bind coordinate location";

  m_uniform_colour = uniformLocation("colour");
  if (m_uniform_colour == -1)
    LOG_ERROR << "V330GLFlatShader2D: Failed to bind colour";

  m_uniform_mvp = uniformLocation("mvp");
  if (m_uniform_mvp == -1)
    LOG_ERROR << "V330GLFlatShader2D: Failed to bind transform";
}

GLFlatShader2D::~GLFlatShader2D() {}

void
GLFlatShader2D::enableCoords()
{
  enableAttributeArray(m_attr_coords);
}

void
GLFlatShader2D::disableCoords()
{
  disableAttributeArray(m_attr_coords);
}

void
GLFlatShader2D::setCoords(const GLfloat* offset, int tupleSize, int stride)
{
  setAttributeArray(m_attr_coords, offset, tupleSize, stride);
  check_gl("Set flatcoords");
}

void
GLFlatShader2D::setCoords(GLuint coords, const GLfloat* offset, int tupleSize, int stride)
{
  glBindBuffer(GL_ARRAY_BUFFER, coords);
  setCoords(offset, tupleSize, stride);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void
GLFlatShader2D::setColour(const glm::vec4& colour)
{
  glUniform4fv(m_uniform_colour, 1, glm::value_ptr(colour));
  check_gl("Set flat uniform colour");
}

void
GLFlatShader2D::setModelViewProjection(const glm::mat4& mvp)
{
  glUniformMatrix4fv(m_uniform_mvp, 1, GL_FALSE, glm::value_ptr(mvp));
  check_gl("Set flat uniform mvp");
}
