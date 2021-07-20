
#include "gl/v33/V33Image3D.h"

#include "Fuse.h"
#include "ImageXYZC.h"
#include "Logging.h"
#include "RenderSettings.h"
#include "gl/Util.h"

#include <algorithm>
#include <array>
#include <iostream>

Image3Dv33::Image3Dv33(std::shared_ptr<ImageXYZC> img)
  : m_vertices(0)
  , m_image_vertices(0)
  , m_image_elements(0)
  , m_num_image_elements(0)
  , m_textureid(0)
  , m_img(img)
  , m_image3d_shader(new GLBasicVolumeShader())
  , m_fusedrgbvolume(nullptr)
{}

Image3Dv33::~Image3Dv33()
{
  delete[] m_fusedrgbvolume;

  glDeleteTextures(1, &m_textureid);
  delete m_image3d_shader;
}

void
Image3Dv33::create()
{
  m_fusedrgbvolume = new uint8_t[3 * m_img->sizeX() * m_img->sizeY() * m_img->sizeZ()];
  // destroy old
  glDeleteTextures(1, &m_textureid);
  // Create image texture.
  glGenTextures(1, &m_textureid);

  setSize(glm::vec2(-(m_img->sizeX() / 2.0f), m_img->sizeX() / 2.0f),
          glm::vec2(-(m_img->sizeY() / 2.0f), m_img->sizeY() / 2.0f));

  // HiLo
  uint8_t lut[256][3];
  for (uint16_t i = 0; i < 256; ++i)
    for (uint16_t j = 0; j < 3; ++j) {
      lut[i][j] = (uint8_t)i;
    }
  lut[0][0] = 0;
  lut[0][2] = 0;
  lut[0][2] = 255;
  lut[255][0] = 255;
  lut[255][1] = 0;
  lut[255][2] = 0;

  glTexImage2D(GL_TEXTURE_1D_ARRAY, // target
               0,                   // level, 0 = base, no minimap,
               GL_RGB8,             // internal format
               256,                 // width
               1,                   // height
               0,                   // border
               GL_RGB,              // external format
               GL_UNSIGNED_BYTE,    // external type
               lut);                // LUT data
  check_gl("Texture create");
}

void
Image3Dv33::render(const CCamera& camera,
                   const Scene* scene,
                   const RenderSettings* renderSettings,
                   float devicePixelRatio)
{
  m_image3d_shader->bind();

  m_image3d_shader->GAMMA_MIN = 0.0;
  m_image3d_shader->GAMMA_MAX = 1.0;
  m_image3d_shader->GAMMA_SCALE = 1.3657f;
  m_image3d_shader->BRIGHTNESS = (1.0f - camera.m_Film.m_Exposure) + 1.0f;
  m_image3d_shader->DENSITY = renderSettings->m_RenderSettings.m_DensityScale / 100.0;
  m_image3d_shader->maskAlpha = 1.0;
  m_image3d_shader->BREAK_STEPS = 512;
  // axis aligned clip planes in object space
  m_image3d_shader->AABB_CLIP_MIN = scene->m_roi.GetMinP() - glm::vec3(0.5, 0.5, 0.5);
  m_image3d_shader->AABB_CLIP_MAX = scene->m_roi.GetMaxP() - glm::vec3(0.5, 0.5, 0.5);
  m_image3d_shader->resolution =
    glm::vec2(camera.m_Film.GetWidth() * devicePixelRatio, camera.m_Film.GetHeight() * devicePixelRatio);
  m_image3d_shader->isPerspective = (camera.m_Projection == PERSPECTIVE) ? 1.0f : 0.0f;
  m_image3d_shader->orthoScale = camera.m_OrthoScale;
  m_image3d_shader->setShadingUniforms();

  // move the box to match where the camera is pointed
  // transform the box from -0.5..0.5 to 0..physicalsize
  glm::vec3 dims(m_img->sizeX() * m_img->physicalSizeX(),
                 m_img->sizeY() * m_img->physicalSizeY(),
                 m_img->sizeZ() * m_img->physicalSizeZ());
  float maxd = (std::max)(dims.x, (std::max)(dims.y, dims.z));
  glm::vec3 scales(dims.x / maxd, dims.y / maxd, dims.z / maxd);
  // it helps to imagine these transforming the space in reverse order
  // (first translate by 0.5, and then scale)
  glm::mat4 mm = glm::scale(glm::mat4(1.0f), scales);
  mm = glm::translate(mm, glm::vec3(0.5, 0.5, 0.5));

  m_image3d_shader->setTransformUniforms(camera, mm);

  glActiveTexture(GL_TEXTURE0);
  check_gl("Activate texture");
  glBindTexture(GL_TEXTURE_3D, m_textureid);
  check_gl("Bind texture");
  m_image3d_shader->setTexture(0);

  glBindVertexArray(m_vertices);

  m_image3d_shader->enableCoords();
  m_image3d_shader->setCoords(m_image_vertices, 0, 3);

  // Push each element to the vertex shader
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_image_elements);
  glDrawElements(GL_TRIANGLES, (GLsizei)m_num_image_elements, GL_UNSIGNED_SHORT, 0);
  check_gl("Image3Dv33 draw elements");

  m_image3d_shader->disableCoords();
  glBindVertexArray(0);

  m_image3d_shader->release();
}

void
Image3Dv33::setSize(const glm::vec2& xlim, const glm::vec2& ylim)
{
  const std::array<GLfloat, 3 * 4 * 2> cube_vertices{
    // front
    -0.5,
    -0.5,
    0.5,
    0.5,
    -0.5,
    0.5,
    0.5,
    0.5,
    0.5,
    -0.5,
    0.5,
    0.5,
    // back
    -0.5,
    -0.5,
    -0.5,
    0.5,
    -0.5,
    -0.5,
    0.5,
    0.5,
    -0.5,
    -0.5,
    0.5,
    -0.5,
  };

  if (m_vertices == 0) {
    glGenVertexArrays(1, &m_vertices);
  }
  glBindVertexArray(m_vertices);

  if (m_image_vertices == 0) {
    glGenBuffers(1, &m_image_vertices);
  }
  glBindBuffer(GL_ARRAY_BUFFER, m_image_vertices);
  glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * cube_vertices.size(), cube_vertices.data(), GL_STATIC_DRAW);

  // note every face of the cube is on a single line
  std::array<GLushort, 36> cube_indices = {
    // front
    0,
    1,
    2,
    2,
    3,
    0,
    // top
    1,
    5,
    6,
    6,
    2,
    1,
    // back
    7,
    6,
    5,
    5,
    4,
    7,
    // bottom
    4,
    0,
    3,
    3,
    7,
    4,
    // left
    4,
    5,
    1,
    1,
    0,
    4,
    // right
    3,
    2,
    6,
    6,
    7,
    3,
  };

  if (m_image_elements == 0) {
    glGenBuffers(1, &m_image_elements);
  }
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_image_elements);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort) * cube_indices.size(), cube_indices.data(), GL_STATIC_DRAW);
  m_num_image_elements = cube_indices.size();
}

void
Image3Dv33::prepareTexture(Scene& s)
{
  auto startTime = std::chrono::high_resolution_clock::now();

  std::vector<glm::vec3> colors;
  for (int i = 0; i < MAX_CPU_CHANNELS; ++i) {
    if (s.m_material.m_enabled[i]) {
      colors.push_back(
        glm::vec3(s.m_material.m_diffuse[i * 3], s.m_material.m_diffuse[i * 3 + 1], s.m_material.m_diffuse[i * 3 + 2]) *
        s.m_material.m_opacity[i]);
    } else {
      colors.push_back(glm::vec3(0, 0, 0));
    }
  }

  Fuse::fuse(m_img.get(), colors, &m_fusedrgbvolume, nullptr);

  auto endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = endTime - startTime;
  LOG_DEBUG << "fuse operation: " << (elapsed.count() * 1000.0) << "ms";
  startTime = std::chrono::high_resolution_clock::now();

  // destroy old
  // glDeleteTextures(1, &_textureid);

  // Create image texture.
  // glGenTextures(1, &_textureid);
  glBindTexture(GL_TEXTURE_3D, m_textureid);
  check_gl("Bind texture");
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  check_gl("Set texture min filter");
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  check_gl("Set texture mag filter");
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  check_gl("Set texture wrap s");
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  check_gl("Set texture wrap t");
  glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
  check_gl("Set texture wrap r");

  GLenum internal_format = GL_RGBA8;
  GLenum external_type = GL_UNSIGNED_BYTE;
  GLenum external_format = GL_RGB;

  // pixel data is tightly packed
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  glTexImage3D(GL_TEXTURE_3D,           // target
               0,                       // level, 0 = base, no minimap,
               internal_format,         // internal format
               (GLsizei)m_img->sizeX(), // width
               (GLsizei)m_img->sizeY(), // height
               (GLsizei)m_img->sizeZ(),
               0,               // border
               external_format, // external format
               external_type,   // external type
               m_fusedrgbvolume);
  check_gl("Volume Texture create");
  //	glGenerateMipmap(GL_TEXTURE_3D);

  endTime = std::chrono::high_resolution_clock::now();
  elapsed = endTime - startTime;
  LOG_DEBUG << "prepare fused 3d rgb texture in " << (elapsed.count() * 1000.0) << "ms";
}
