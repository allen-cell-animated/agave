#pragma once

#include "glad/glad.h"

#include "gl/Util.h"

#include <glm.h>

/**
 * 2D line shader program.
 */
class GLLineShader2D : public GLShaderProgram
{

public:
  /**
   * Constructor.
   *
   * @param parent the parent of this object.
   */
  explicit GLLineShader2D();

  /// Destructor.
  ~GLLineShader2D();

  /// @copydoc GLImageShader2D::enableCoords()
  void enableCoords();

  /// @copydoc GLImageShader2D::enableCoords()
  void disableCoords();

  /// @copydoc GLImageShader2D::setCoords(const GLfloat*, int, int)
  void setCoords(const GLfloat* offset, int tupleSize, int stride = 0);

  /// @copydoc GLImageShader2D::setCoords(QOpenGLBuffer&, const GLfloat *, int, int)
  void setCoords(GLuint coords, const GLfloat* offset, int tupleSize, int stride = 0);

  /// Enable colour array.
  void enableColour();

  /// Disable colour array.
  void disableColour();

  /**
   * Set colours from array.
   *
   * @param offset data offset if using a buffer object otherwise
   * the colour values.
   * @param tupleSize the tuple size of the data.
   * @param stride the stride of the data.
   */
  void setColour(const GLfloat* offset, int tupleSize, int stride = 0);

  /**
   * Set colours from buffer object.
   *
   * @param colours the colour values; null if using a buffer
   * object.
   * @param offset the offset into the colours buffer.
   * @param tupleSize the tuple size of the data.
   * @param stride the stride of the data.
   */
  void setColour(GLuint colours, const GLfloat* offset, int tupleSize, int stride = 0);

  /// @copydoc GLImageShader2D::setModelViewProjection(const glm::mat4& mvp)
  void setModelViewProjection(const glm::mat4& mvp);

  /**
   * Set zoom level.
   *
   * @param zoom the zoom level.
   */
  void setZoom(float zoom);

private:
  /// @copydoc GLImageShader2D::vshader
  GLShader* m_vshader;
  /// @copydoc GLImageShader2D::fshader
  GLShader* m_fshader;

  /// @copydoc GLImageShader2D::attr_coords
  int m_attr_coords;
  /// Vertex colour attribute
  int m_attr_colour;
  /// @copydoc GLImageShader2D::uniform_mvp
  int m_uniform_mvp;
  /// Zoom uniform.
  int m_uniform_zoom;
};
