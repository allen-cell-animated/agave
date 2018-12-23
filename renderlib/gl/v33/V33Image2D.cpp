#include "gl/v33/V33Image2D.h"
#include "gl/Util.h"

#include <iostream>

Image2Dv33::Image2Dv33(std::shared_ptr<ImageXYZC> img)
  : Image2D(img)
  , m_image_shader(new GLImageShader2D())
{}

Image2Dv33::~Image2Dv33()
{
  delete m_image_shader;
}

void
Image2Dv33::render(const glm::mat4& mvp)
{
  m_image_shader->bind();

  m_image_shader->setMin(texmin);
  m_image_shader->setMax(texmax);
  m_image_shader->setCorrection(texcorr);
  m_image_shader->setModelViewProjection(mvp);

  glActiveTexture(GL_TEXTURE0);
  check_gl("Activate texture");
  glBindTexture(GL_TEXTURE_2D, textureid);
  check_gl("Bind texture");
  m_image_shader->setTexture(0);

  glActiveTexture(GL_TEXTURE1);
  check_gl("Activate texture");
  glBindTexture(GL_TEXTURE_1D_ARRAY, lutid);
  check_gl("Bind texture");
  m_image_shader->setLUT(1);

  glBindVertexArray(vertices);

  m_image_shader->enableCoords();
  m_image_shader->setCoords(image_vertices, 0, 2);

  m_image_shader->enableTexCoords();
  m_image_shader->setTexCoords(image_texcoords, 0, 2);

  // Push each element to the vertex shader
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, image_elements);
  glDrawElements(GL_TRIANGLES, (GLsizei)num_image_elements, GL_UNSIGNED_SHORT, 0);
  check_gl("Image2D draw elements");

  m_image_shader->disableCoords();
  m_image_shader->disableTexCoords();
  glBindVertexArray(0);

  m_image_shader->release();
}
