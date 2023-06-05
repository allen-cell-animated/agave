#pragma once

#include "gl/Util.h"

#include <glm.h>

/**
 */
class GLToneMapShader : public GLShaderProgram
{

public:
  /**
   * Constructor.
   *
   * @param parent the parent of this object.
   */
  explicit GLToneMapShader();

  /// Destructor.
  ~GLToneMapShader();

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

  void setShadingUniforms(float invExposure);

private:
  /// The vertex shader.
  GLShader* m_vshader;
  /// The fragment shader.
  GLShader* m_fshader;

  int m_texture;
  int m_InvExposure;
};
