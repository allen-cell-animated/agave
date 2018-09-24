#include "gl/v33/V33FSQ.h"
#include "gl/Util.h"

#include <iostream>


FSQ::FSQ(QOpenGLShaderProgram * shdr):
    Image2D(),
    image_shader(shdr)
{
    // shder must have a position attr
    attr_coords = image_shader ? image_shader->attributeLocation("position") : 0;
}

FSQ::~FSQ()
{
	delete image_shader;
}

void
FSQ::render(const glm::mat4& mvp)
{
    if (!image_shader) {
        return;
    }
	glBindVertexArray(vertices);

    image_shader->bind();

    image_shader->enableAttributeArray(attr_coords);
    image_shader->setAttributeArray(attr_coords, image_vertices, 0, 2);

    // Push each element to the vertex shader
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, image_elements);
    glDrawElements(GL_TRIANGLES, (GLsizei)num_image_elements, GL_UNSIGNED_SHORT, 0);
    check_gl("Image2D draw elements");

    image_shader->disableAttributeArray(attr_coords);
    glBindVertexArray(0);

	image_shader->release();
}

