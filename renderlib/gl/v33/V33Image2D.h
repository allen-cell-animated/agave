#pragma once

#include "gl/Image2D.h"
#include "glsl/v330/V330GLImageShader2D.h"
#include "glsl/v330/GLBasicVolumeShader.h"


/**
    * 2D (xy) image renderer.
    *
    * Draws the specified image, using a user-selectable plane.
    *
    * The render is greyscale with a per-channel min/max for linear
    * contrast.
    */
class Image2Dv33 : public Image2D
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
    explicit Image2Dv33(std::shared_ptr<ImageXYZC>  img);

    /// Destructor.
    virtual ~Image2Dv33();

	void
		render(const glm::mat4& mvp);

private:
    /// The shader program for image rendering.
	GLImageShader2D *m_image_shader;
};

