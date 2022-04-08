#pragma once

#include "glad/glad.h"

#include "gl/Util.h"

#include <glm.h>

/**
 * 2D image shader program (simple, up to three channels).
 */
class GLImageShader2D : public GLShaderProgram
{

public:
  /**
   * Constructor.
   *
   * @param parent the parent of this object.
   */
  explicit GLImageShader2D();

  /// Destructor.
  ~GLImageShader2D();

  /// Enable vertex coordinates.
  void enableCoords();

  /// Disable vertex coordinates.
  void disableCoords();

  /**
   * Set vertex coordinates from array.
   *
   * @param offset data offset if using a buffer object otherwise
   * the coordinate values.
   * @param tupleSize the tuple size of the data.
   * @param stride the stride of the data.
   */
  void setCoords(const GLfloat* offset = 0, int tupleSize = 2, int stride = 0);

  /**
   * Set vertex coordinates from buffer object.
   *
   * @param coords the coordinate values; null if using a buffer object.
   * @param offset the offset into the coords buffer.
   * @param tupleSize the tuple size of the data.
   * @param stride the stride of the data.
   */
  void setCoords(GLuint coords, const GLfloat* offset = 0, int tupleSize = 2, int stride = 0);

  /// Enable texture coordinates.
  void enableTexCoords();

  /// Disable texture coordinates.
  void disableTexCoords();

  /**
   * Set texture coordinates from array.
   *
   * @param offset data offset if using a buffer object otherwise
   * the coordinate values.
   * @param tupleSize the tuple size of the data.
   * @param stride the stride of the data.
   */
  void setTexCoords(const GLfloat* offset = 0, int tupleSize = 2, int stride = 0);

  /**
   * Set texture coordinates from buffer object.
   *
   * @param coords the coordinate values; null if using a buffer
   * object.
   * @param offset the offset into the coords buffer.
   * @param tupleSize the tuple size of the data.
   * @param stride the stride of the data.
   */
  void setTexCoords(GLuint coords, const GLfloat* offset = 0, int tupleSize = 2, int stride = 0);

  /**
   * Set the texture to render.
   *
   * @param texunit the texture unit to use.
   */
  void setTexture(int texunit);

  /**
   * Set minimum limits for linear contrast.
   *
   * @param min the RGB channel limits.
   */
  void setMin(const glm::vec3& min);

  /**
   * Set maximum limits for linear contrast.
   *
   * @param max the RGB channel limits.
   */
  void setMax(const glm::vec3& max);

  /**
   * Set correction multipliers to normalise pixel intensity.
   *
   * Use to correct the pixel value limits to the storage size
   * limits, for example when using data with 12 bits per
   * sample with a 16-bit storage type it will require
   * multiplying by 2^(16-12) = 2^4 = 16.  To leave
   * uncorrected, e.g. for float and complex types, and
   * integer types where the bits per sample is the same as
   * the storage size, set to 1.0.
   *
   * @param corr the RGB channel correction multipliers.
   */
  void setCorrection(const glm::vec3& corr);

  /**
   * Set the LUT to use.
   *
   * @param texunit the texture unit to use.
   */
  void setLUT(int texunit);

  /**
   * Set model view projection matrix.
   *
   * @param mvp the model view projection matrix.
   */
  void setModelViewProjection(const glm::mat4& mvp);

private:
  /// The vertex shader.
  GLShader* m_vshader;
  /// The fragment shader.
  GLShader* m_fshader;

  /// Vertex coordinates attribute.
  int m_attr_coords;
  /// Texture coordinates attribute.
  int m_attr_texcoords;
  /// Model view projection uniform.
  int m_uniform_mvp;
  /// Texture uniform.
  int m_uniform_texture;
  /// LUT uniform.
  int m_uniform_lut;
  /// Minimum limits for linear contrast uniform.
  int m_uniform_min;
  /// Maximum limits for linear contrast uniform.
  int m_uniform_max;
  /// Correction multiplier for linear contrast uniform.
  int m_uniform_corr;
};
