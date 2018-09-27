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
    destroy();
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

    check_gl("Image2D bound buffers");
    image_shader->enableAttributeArray(attr_coords);
    static const GLfloat square_vertices_a[8] = { -1, -1,
        1, -1,
        1, 1,
        -1, 1 };

    glBindBuffer(GL_ARRAY_BUFFER, image_vertices);

    image_shader->setAttributeArray(attr_coords, 0, 2, 0);
    //setCoords(offset, tupleSize, stride);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Push each element to the vertex shader
    check_gl("Image2D attribute array");

    glDrawElements(GL_TRIANGLES, (GLsizei)num_image_elements, GL_UNSIGNED_SHORT, 0);
    check_gl("Image2D draw elements");

    glBindVertexArray(0);

	image_shader->release();
}

