#include "V330GLLineShader2D.h"

#include "Logging.h"

#include <gl/Util.h>
#include <glm.h>

#include <iostream>

GLLineShader2D::GLLineShader2D()
  : GLShaderProgram()
  , m_vshader()
  , m_fshader()
  , m_attr_coords()
  , m_attr_colour()
  , m_uniform_mvp()
{
  m_vshader = new GLShader(GL_VERTEX_SHADER);
  m_vshader->compileSourceCode(
    "#version 400 core\n"
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
  if (!m_vshader->isCompiled()) {
    LOG_ERROR << "GLLineShader2D: Failed to compile vertex shader\n" << m_vshader->log();
  }

  m_fshader = new GLShader(GL_FRAGMENT_SHADER);
  m_fshader->compileSourceCode("#version 400 core\n"
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
    LOG_ERROR << "GLLineShader2D: Failed to compile fragment shader\n" << m_fshader->log();
  }

  addShader(m_vshader);
  addShader(m_fshader);
  link();

  if (!isLinked()) {
    LOG_ERROR << "GLLineShader2D: Failed to link shader program\n" << log();
  }

  m_attr_coords = attributeLocation("coord2d");
  if (m_attr_coords == -1)
    LOG_ERROR << "GLLineShader2D: Failed to bind coordinate location";

  m_attr_colour = attributeLocation("colour");
  if (m_attr_coords == -1)
    LOG_ERROR << "GLLineShader2D: Failed to bind colour location";

  m_uniform_mvp = uniformLocation("mvp");
  if (m_uniform_mvp == -1)
    LOG_ERROR << "GLLineShader2D: Failed to bind transform";

  m_uniform_zoom = uniformLocation("zoom");
  if (m_uniform_zoom == -1)
    LOG_ERROR << "GLLineShader2D: Failed to bind zoom factor";
}

GLLineShader2D::~GLLineShader2D() {}

void
GLLineShader2D::enableCoords()
{
  enableAttributeArray(m_attr_coords);
}

void
GLLineShader2D::disableCoords()
{
  disableAttributeArray(m_attr_coords);
}

void
GLLineShader2D::setCoords(const GLfloat* offset, int tupleSize, int stride)
{
  setAttributeArray(m_attr_coords, offset, tupleSize, stride);
}

void
GLLineShader2D::setCoords(GLuint coords, const GLfloat* offset, int tupleSize, int stride)
{
  glBindBuffer(GL_ARRAY_BUFFER, coords);
  setCoords(offset, tupleSize, stride);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void
GLLineShader2D::enableColour()
{
  enableAttributeArray(m_attr_colour);
}

void
GLLineShader2D::disableColour()
{
  disableAttributeArray(m_attr_colour);
}

void
GLLineShader2D::setColour(const GLfloat* offset, int tupleSize, int stride)
{
  setAttributeArray(m_attr_colour, offset, tupleSize, stride);
}

void
GLLineShader2D::setColour(GLuint colour, const GLfloat* offset, int tupleSize, int stride)
{
  glBindBuffer(GL_ARRAY_BUFFER, colour);
  setColour(offset, tupleSize, stride);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void
GLLineShader2D::setModelViewProjection(const glm::mat4& mvp)
{
  glUniformMatrix4fv(m_uniform_mvp, 1, GL_FALSE, glm::value_ptr(mvp));
  check_gl("Set line uniform mvp");
}

void
GLLineShader2D::setZoom(float zoom)
{
  glUniform1f(m_uniform_zoom, zoom);
  check_gl("Set line zoom level");
}
