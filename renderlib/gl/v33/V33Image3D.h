#pragma once

#include "glsl/v330/GLBasicVolumeShader.h"

#include <memory>

class ImageXYZC;

/**
    * 2D (xy) image renderer.
    *
    * Draws the specified image, using a user-selectable plane.
    *
    * The render is greyscale with a per-channel min/max for linear
    * contrast.
    */
class Image3Dv33
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
    explicit Image3Dv33(std::shared_ptr<ImageXYZC>  img);

    /// Destructor.
    virtual ~Image3Dv33();

	void create();

	void
		render(const Camera& camera);


	/**
	* Get minimum limit for linear contrast.
	*
	* @returns the limits for three channels.
	*/
	const glm::vec3&
		getMin() const;

	/**
	* Set minimum limit for linear contrast.
	*
	* Note that depending upon the image type, not all channels may
	* be used.
	*
	* @param min the limits for three channels.
	*/
	void
		setMin(const glm::vec3& min) {}

	/**
	* Get maximum limit for linear contrast.
	*
	* @returns the limits for three channels.
	*/
	const glm::vec3&
		getMax() const;

	/**
	* Set maximum limit for linear contrast.
	*
	* Note that depending upon the image type, not all channels may
	* be used.
	*
	* @param max the limits for three channels.
	*/
	void
		setMax(const glm::vec3& max) {}

	void setPlane(int plane, int z, int c);

	/**
	* Get texture ID.
	*
	* This is the identifier of the texture for the plane being
	* rendered.
	*
	* @returns the texture ID.
	*/
	unsigned int
		texture();

	/**
	* Get LUT ID.
	*
	* This is the identifier of the LUT for the plane being
	* rendered.
	*
	* @returns the LUT ID.
	*/
	unsigned int
		lut();

protected:
	/**
	* Set the size of the x and y dimensions.
	*
	* @param xlim the x axis limits (range).
	* @param ylim the y axis limits (range).
	*/
	virtual
		void
		setSize(const glm::vec2& xlim,
			const glm::vec2& ylim);

	/// The vertex array.
	GLuint vertices;  // vao
	/// The image vertices.
	GLuint image_vertices;  // buffer
	/// The image elements.
	GLuint image_elements;  // buffer
	size_t num_image_elements;
	/// The identifier of the texture owned and used by this object.
	unsigned int textureid;
	/// The identifier of the LUTs owned and used by this object.
	unsigned int lutid;
	/// Linear contrast minimum limits.
	float texmin;
	/// Linear contrast maximum limits.
	float texmax;
	/// Linear contrast correction multipliers.
	//glm::vec3 texcorr;
	/// The image wrapped as a flat data ptr
	std::shared_ptr<ImageXYZC> _img;
	int _c;

private:
    /// The shader program for image rendering.
	GLBasicVolumeShader *image3d_shader;
};

