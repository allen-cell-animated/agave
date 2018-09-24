#pragma once

#include "gl/Image2D.h"

#include <QOpenGLShaderProgram>

/**
    * 2D (xy) image renderer.
    *
    * Draws the specified image, using a user-selectable plane.
    *
    * The render is greyscale with a per-channel min/max for linear
    * contrast.
    */
class FSQ : public Image2D
{

public:
    /**
    * Create a 2D image.
    *
    * The size and position will be taken from the specified image.
    *
    * @param reader the image reader.
    * @param series the image series.
    * @param parent the parent of this object.
    */
    explicit FSQ(QOpenGLShaderProgram * shdr);

    /// Destructor.
    virtual ~FSQ();

	void
		render(const glm::mat4& mvp);

private:
    /// The shader program for image rendering.
    QOpenGLShaderProgram *image_shader;
    int attr_coords;

};

