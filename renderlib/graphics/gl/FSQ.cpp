#include "gl/FSQ.h"
#include "gl/Util.h"

#include <iostream>

FSQ::FSQ()
  : Image2D()
{
}

FSQ::~FSQ()
{
  destroy();
}

void
FSQ::render(const glm::mat4& mvp)
{
  glBindVertexArray(m_vertices);
  check_gl("FSQ bound buffers");

  glDrawElements(GL_TRIANGLES, (GLsizei)m_num_image_elements, GL_UNSIGNED_SHORT, 0);
  check_gl("FSQ draw elements");

  glBindVertexArray(0);
}
