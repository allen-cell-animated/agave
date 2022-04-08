#include "V330GLImageShader2D.h"

#include "Logging.h"

#include <gl/Util.h>
#include <glm.h>

#include <iostream>
#include <sstream>

GLImageShader2D::GLImageShader2D()
  : GLShaderProgram()
  , m_vshader()
  , m_fshader()
  , m_attr_coords()
  , m_attr_texcoords()
  , m_uniform_mvp()
  , m_uniform_texture()
  , m_uniform_lut()
  , m_uniform_min()
  , m_uniform_max()
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
    LOG_ERROR << "V330GLImageShader2D: Failed to compile vertex shader\n" << m_vshader->log();
  }

  m_fshader = new GLShader(GL_FRAGMENT_SHADER);
  m_fshader->compileSourceCode("#version 400 core\n"
                               "\n"
                               "uniform sampler2D tex;\n"
                               "uniform sampler1DArray lut;\n"
                               "uniform vec3 texmin;\n"
                               "uniform vec3 texmax;\n"
                               "uniform vec3 correction;\n"
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
                               "  outputColour = texture(lut, vec2(((((texval[0] * correction[0]) - texmin[0]) / "
                               "(texmax[0] - texmin[0]))), 0.0));\n"
                               "}\n");

  if (!m_fshader->isCompiled()) {
    LOG_ERROR << "V330GLImageShader2D: Failed to compile fragment shader\n" << m_fshader->log();
  }

  addShader(m_vshader);
  addShader(m_fshader);
  link();

  if (!isLinked()) {
    LOG_ERROR << "V330GLImageShader2D: Failed to link shader program\n" << log();
  }

  m_attr_coords = attributeLocation("coord2d");
  if (m_attr_coords == -1)
    LOG_ERROR << "V330GLImageShader2D: Failed to bind coordinates";

  m_attr_texcoords = attributeLocation("texcoord");
  if (m_attr_texcoords == -1)
    LOG_ERROR << "V330GLImageShader2D: Failed to bind texture coordinates";

  m_uniform_mvp = uniformLocation("mvp");
  if (m_uniform_mvp == -1)
    LOG_ERROR << "V330GLImageShader2D: Failed to bind transform";

  m_uniform_texture = uniformLocation("tex");
  if (m_uniform_texture == -1)
    LOG_ERROR << "V330GLImageShader2D: Failed to bind texture uniform ";

  m_uniform_lut = uniformLocation("lut");
  if (m_uniform_lut == -1)
    LOG_ERROR << "V330GLImageShader2D: Failed to bind lut uniform ";

  m_uniform_min = uniformLocation("texmin");
  if (m_uniform_min == -1)
    LOG_ERROR << "V330GLImageShader2D: Failed to bind min uniform ";

  m_uniform_max = uniformLocation("texmax");
  if (m_uniform_max == -1)
    LOG_ERROR << "V330GLImageShader2D: Failed to bind max uniform ";

  m_uniform_corr = uniformLocation("correction");
  if (m_uniform_corr == -1)
    LOG_ERROR << "V330GLImageShader2D: Failed to bind correction uniform ";
}

GLImageShader2D::~GLImageShader2D() {}

void
GLImageShader2D::enableCoords()
{
  enableAttributeArray(m_attr_coords);
}

void
GLImageShader2D::disableCoords()
{
  disableAttributeArray(m_attr_coords);
}

void
GLImageShader2D::setCoords(const GLfloat* offset, int tupleSize, int stride)
{
  setAttributeArray(m_attr_coords, offset, tupleSize, stride);
}

void
GLImageShader2D::setCoords(GLuint coords, const GLfloat* offset, int tupleSize, int stride)
{
  glBindBuffer(GL_ARRAY_BUFFER, coords);
  setCoords(offset, tupleSize, stride);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void
GLImageShader2D::enableTexCoords()
{
  enableAttributeArray(m_attr_texcoords);
}

void
GLImageShader2D::disableTexCoords()
{
  disableAttributeArray(m_attr_texcoords);
}

void
GLImageShader2D::setTexCoords(const GLfloat* offset, int tupleSize, int stride)
{
  setAttributeArray(m_attr_texcoords, offset, tupleSize, stride);
}

void
GLImageShader2D::setTexCoords(GLuint texcoords, const GLfloat* offset, int tupleSize, int stride)
{
  glBindBuffer(GL_ARRAY_BUFFER, texcoords);
  setTexCoords(offset, tupleSize, stride);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void
GLImageShader2D::setTexture(int texunit)
{
  glUniform1i(m_uniform_texture, texunit);
  check_gl("Set image texture");
}

void
GLImageShader2D::setMin(const glm::vec3& min)
{
  glUniform3fv(m_uniform_min, 1, glm::value_ptr(min));
  check_gl("Set min range");
}

void
GLImageShader2D::setMax(const glm::vec3& max)
{
  glUniform3fv(m_uniform_max, 1, glm::value_ptr(max));
  check_gl("Set max range");
}

void
GLImageShader2D::setCorrection(const glm::vec3& corr)
{
  glUniform3fv(m_uniform_corr, 1, glm::value_ptr(corr));
  check_gl("Set correction multiplier");
}

void
GLImageShader2D::setLUT(int texunit)
{
  glUniform1i(m_uniform_lut, texunit);
  check_gl("Set LUT texture");
}

void
GLImageShader2D::setModelViewProjection(const glm::mat4& mvp)
{
  glUniformMatrix4fv(m_uniform_mvp, 1, GL_FALSE, glm::value_ptr(mvp));
  check_gl("Set image2d uniform mvp");
}
