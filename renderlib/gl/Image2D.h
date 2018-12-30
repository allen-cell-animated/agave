#pragma once

#include <memory>

#include <glad/glad.h>

#include "glm.h"


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
  Image2D();

  /// Destructor.
  virtual ~Image2D() = 0;

  /**
   * Create GL buffers.
   *
   * @note Requires a valid GL context.  Must be called before
   * rendering.
   */
  virtual void create();

  /**
   * Set the size of the x and y dimensions.
   *
   * @param xlim the x axis limits (range).
   * @param ylim the y axis limits (range).
   */
  virtual void setSize(const glm::vec2& xlim, const glm::vec2& ylim);

public:

  /**
   * Get minimum limit for linear contrast.
   *
   * @returns the limits for three channels.
   */
  const glm::vec3& getMin() const;

  /**
   * Set minimum limit for linear contrast.
   *
   * Note that depending upon the image type, not all channels may
   * be used.
   *
   * @param min the limits for three channels.
   */
  void setMin(const glm::vec3& min);

  /**
   * Get maximum limit for linear contrast.
   *
   * @returns the limits for three channels.
   */
  const glm::vec3& getMax() const;

  /**
   * Set maximum limit for linear contrast.
   *
   * Note that depending upon the image type, not all channels may
   * be used.
   *
   * @param max the limits for three channels.
   */
  void setMax(const glm::vec3& max);

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
  virtual void render(const glm::mat4& mvp) = 0;

  /**
   * Get texture ID.
   *
   * This is the identifier of the texture for the plane being
   * rendered.
   *
   * @returns the texture ID.
   */
  unsigned int texture();

  /**
   * Get LUT ID.
   *
   * This is the identifier of the LUT for the plane being
   * rendered.
   *
   * @returns the LUT ID.
   */
  unsigned int lut();

protected:
  void destroy();

  /// The vertex array.
  GLuint m_vertices; // vao
  /// The image vertices.
  GLuint m_image_vertices; // buffer
  /// The image texture coordinates.
  GLuint m_image_texcoords; // buffer
  /// The image elements.
  GLuint m_image_elements; // buffer
  size_t m_num_image_elements;
  /// The identifier of the texture owned and used by this object.
  unsigned int m_textureid;
  /// The identifier of the LUTs owned and used by this object.
  unsigned int m_lutid;
  /// Linear contrast minimum limits.
  glm::vec3 m_texmin;
  /// Linear contrast maximum limits.
  glm::vec3 m_texmax;
  /// Linear contrast correction multipliers.
  glm::vec3 m_texcorr;
};
