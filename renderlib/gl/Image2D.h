#pragma once

#include <memory>

#include "glm.h"

#include "glad/include/glad/glad.h"

class ImageXYZC;

/**
* 2D (xy) image renderer.
*
* Draws the specified image, using a user-selectable plane.
*
* The render is greyscale with a per-channel min/max for linear
* contrast.
*/
class Image2D
{
public:
    explicit
        Image2D();

    
    /**
		* Create a 2D image.
		*
		* The size and position will be taken from the specified image.
		*
		* @param reader the image reader.
		* @param series the image series.
		* @param parent the parent of this object.
		*/
	explicit
	Image2D(std::shared_ptr<ImageXYZC>  img);

	/// Destructor.
	virtual
	~Image2D() = 0;

	/**
		* Create GL buffers.
		*
		* @note Requires a valid GL context.  Must be called before
		* rendering.
		*/
	virtual
	void
	create();

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

	public:
	/**
		* Set the plane to render.
		*
		* @param plane the plane number.
		*/
	void
	setPlane(size_t plane, size_t z, size_t c);

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
	setMin(const glm::vec3& min);

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
	setMax(const glm::vec3& max);

	/**
		* Range of min/max adjustment for linear contrast.
		*/
	enum RangePolicy
		{
		StorageRange, ///< Range of storage type.
		BPPRange,     ///< Range of pixel type and bits per pixel.
		PlaneRange,   ///< Range of samples on the current plane.
		ImageRange    ///< Range of samples in the current image.
		};

	/**
		* Render the image.
		*
		* @param mvp the model view projection matrix.
		*/
	virtual
	void
	render(const glm::mat4& mvp) = 0;

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
	/// The vertex array.
	GLuint vertices;  // vao
	/// The image vertices.
	GLuint image_vertices;  // buffer
	/// The image texture coordinates.
	GLuint image_texcoords; // buffer
	/// The image elements.
	GLuint image_elements;  // buffer
	size_t num_image_elements;
	/// The identifier of the texture owned and used by this object.
	unsigned int textureid;
	/// The identifier of the LUTs owned and used by this object.
	unsigned int lutid;
	/// Linear contrast minimum limits.
	glm::vec3 texmin;
	/// Linear contrast maximum limits.
	glm::vec3 texmax;
	/// Linear contrast correction multipliers.
	glm::vec3 texcorr;
	/// The image wrapped as a flat data ptr
	std::shared_ptr<ImageXYZC> _img;
	/// The current image plane.
	size_t plane;
};
