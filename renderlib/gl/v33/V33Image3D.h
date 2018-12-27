#pragma once

#include "AppScene.h"
#include "glsl/v330/GLBasicVolumeShader.h"
#include <memory>

class ImageXYZC;
class RenderSettings;

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
  explicit Image3Dv33(std::shared_ptr<ImageXYZC> img);

  /// Destructor.
  virtual ~Image3Dv33();

  void create();

  void render(const CCamera& camera, const Scene* scene, const RenderSettings* renderSettings);

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
  void setMin(const glm::vec3& min) {}

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
  void setMax(const glm::vec3& max) {}

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

  void prepareTexture(Scene& s);

protected:
  /**
   * Set the size of the x and y dimensions.
   *
   * @param xlim the x axis limits (range).
   * @param ylim the y axis limits (range).
   */
  virtual void setSize(const glm::vec2& xlim, const glm::vec2& ylim);

  /// The vertex array.
  GLuint m_vertices; // vao
  /// The image vertices.
  GLuint m_image_vertices; // buffer
  /// The image elements.
  GLuint m_image_elements; // buffer
  size_t m_num_image_elements;
  /// The identifier of the texture owned and used by this object.
  unsigned int m_textureid;
  /// The identifier of the LUTs owned and used by this object.
  unsigned int m_lutid;
  /// Linear contrast minimum limits.
  float m_texmin;
  /// Linear contrast maximum limits.
  float m_texmax;
  /// Linear contrast correction multipliers.
  // glm::vec3 texcorr;
  /// The image wrapped as a flat data ptr
  std::shared_ptr<ImageXYZC> m_img;
  int m_c;

private:
  /// The shader program for image rendering.
  GLBasicVolumeShader* m_image3d_shader;

  uint8_t* m_fusedrgbvolume;
};
