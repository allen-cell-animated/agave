#include "gl/v33/V33Image2D.h"
#include "gl/Util.h"

#include <iostream>


Image2Dv33::Image2Dv33(std::shared_ptr<ImageXYZC>  img):
    Image2D(img),
    image_shader(new GLImageShader2D())
{
}

Image2Dv33::~Image2Dv33()
{
	delete image_shader;
}

void
Image2Dv33::render(const glm::mat4& mvp)
{
    image_shader->bind();

    image_shader->setMin(texmin);
    image_shader->setMax(texmax);
    image_shader->setCorrection(texcorr);
    image_shader->setModelViewProjection(mvp);

    glActiveTexture(GL_TEXTURE0);
    check_gl("Activate texture");
    glBindTexture(GL_TEXTURE_2D, textureid);
    check_gl("Bind texture");
    image_shader->setTexture(0);

    glActiveTexture(GL_TEXTURE1);
    check_gl("Activate texture");
    glBindTexture(GL_TEXTURE_1D_ARRAY, lutid);
    check_gl("Bind texture");
    image_shader->setLUT(1);

	glBindVertexArray(vertices);

    image_shader->enableCoords();
    image_shader->setCoords(image_vertices, 0, 2);

    image_shader->enableTexCoords();
    image_shader->setTexCoords(image_texcoords, 0, 2);

    // Push each element to the vertex shader
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, image_elements);
    glDrawElements(GL_TRIANGLES, (GLsizei)num_image_elements, GL_UNSIGNED_SHORT, 0);
    check_gl("Image2D draw elements");

    image_shader->disableCoords();
    image_shader->disableTexCoords();
	glBindVertexArray(0);

	image_shader->release();
}

