#include "Image2D.h"
#include "Util.h"

#include "ImageXYZC.h"
#include "glad/glad.h"

#include <array>
#include <iostream>

#if 0
namespace
{
  class TextureProperties
  {
  public:
    GLenum internal_format;
    GLenum external_format;
    GLint external_type;
    bool make_normal;
    GLint min_filter;
    GLint mag_filter;
    ome::files::dimension_size_type w;
    ome::files::dimension_size_type h;

    TextureProperties(const ome::files::FormatReader& reader,
                      ome::files::dimension_size_type series):
      internal_format(GL_R8),
      external_format(GL_RED),
      external_type(GL_UNSIGNED_BYTE),
      make_normal(false),
      min_filter(GL_LINEAR_MIPMAP_LINEAR),
      mag_filter(GL_LINEAR),
      w(0),
      h(0)
    {
      ome::files::dimension_size_type oldseries = reader.getSeries();
      reader.setSeries(series);
      ome::xml::model::enums::PixelType pixeltype = reader.getPixelType();
      reader.setSeries(oldseries);

      w = reader.getSizeX();
      h = reader.getSizeY();

      switch(pixeltype)
        {
        case ::ome::xml::model::enums::PixelType::INT8:
          internal_format = GL_R8;
          external_type = GL_BYTE;
          break;
        case ::ome::xml::model::enums::PixelType::INT16:
          internal_format = GL_R16;
          external_type = GL_SHORT;
          break;
        case ::ome::xml::model::enums::PixelType::INT32:
          internal_format = GL_R16;
          external_type = GL_INT;
          make_normal = true;
          break;
        case ::ome::xml::model::enums::PixelType::UINT8:
          internal_format = GL_R8;
          external_type = GL_UNSIGNED_BYTE;
          break;
        case ::ome::xml::model::enums::PixelType::UINT16:
          internal_format = GL_R16;
          external_type = GL_UNSIGNED_SHORT;
          break;
        case ::ome::xml::model::enums::PixelType::UINT32:
          internal_format = GL_R16;
          external_type = GL_UNSIGNED_INT;
          make_normal = true;
          break;
        case ::ome::xml::model::enums::PixelType::FLOAT:
          internal_format = GL_R32F;
          if (!GL_ARB_texture_float)
            internal_format = GL_R16;
          external_type = GL_FLOAT;
          break;
        case ::ome::xml::model::enums::PixelType::DOUBLE:
          internal_format = GL_R32F;
          if (!GL_ARB_texture_float)
            internal_format = GL_R16;
          external_type = GL_DOUBLE;
          break;
        case ::ome::xml::model::enums::PixelType::BIT:
          internal_format = GL_R8;
          external_type = GL_UNSIGNED_BYTE;
          make_normal = true;
          min_filter = GL_NEAREST_MIPMAP_LINEAR;
          mag_filter = GL_NEAREST;
          break;
        case ::ome::xml::model::enums::PixelType::COMPLEXFLOAT:
          internal_format = GL_RG32F;
          if (!GL_ARB_texture_float)
            internal_format = GL_RG16;
          external_type = GL_FLOAT;
          external_format = GL_RG;
        case ::ome::xml::model::enums::PixelType::COMPLEXDOUBLE:
          internal_format = GL_RG32F;
          if (!GL_ARB_texture_float)
            internal_format = GL_RG16;
          external_type = GL_DOUBLE;
          external_format = GL_RG;
          break;
        }
    }
  };

  /*
   * Assign VariantPixelBuffer to OpenGL texture buffer.
   *
   * The following buffer types are supported:
   * - RGB subchannel, single channel for simple numeric types
   * - no subchannel, single channel for simple numeric types
   * - no subchannel, single channel for complex numeric types
   *
   * The buffer may only contain a single xy plane; no higher
   * dimensions may be used.
   *
   * If OpenGL limitations require
   */
  struct GLSetBufferVisitor : public boost::static_visitor<>
  {
    unsigned int textureid;
    TextureProperties tprop;

    GLSetBufferVisitor(unsigned int textureid,
                       const TextureProperties& tprop):
      textureid(textureid),
      tprop(tprop)
    {
    }

    PixelBufferBase::storage_order_type
    gl_order(const PixelBufferBase::storage_order_type& order)
    {
      PixelBufferBase::storage_order_type ret(order);
      // This makes the assumption that the order is SXY or XYS, and
      // switches XYS to SXY if needed.
      if (order.ordering(0) != ome::files::DIM_SUBCHANNEL)
        {
          PixelBufferBase::size_type ordering[PixelBufferBase::dimensions];
          bool ascending[PixelBufferBase::dimensions] = {true, true, true, true, true, true, true, true, true};
          for (boost::detail::multi_array::size_type d = 0; d < PixelBufferBase::dimensions; ++d)
            {
              ordering[d] = order.ordering(d);
              ascending[d] = order.ascending(d);

              PixelBufferBase::size_type xo = ordering[0];
              PixelBufferBase::size_type yo = ordering[1];
              PixelBufferBase::size_type so = ordering[2];
              bool xa = ascending[0];
              bool ya = ascending[1];
              bool sa = ascending[2];

              ordering[0] = so;
              ordering[1] = xo;
              ordering[2] = yo;
              ascending[0] = sa;
              ascending[1] = xa;
              ascending[2] = ya;

              ret = PixelBufferBase::storage_order_type(ordering, ascending);
            }
        }
      return ret;
    }

    template<typename T>
    void
    operator() (const T& v)
    {
      typedef typename T::element_type::value_type value_type;

      T src_buffer(v);
      const PixelBufferBase::storage_order_type& orig_order(v->storage_order());
      PixelBufferBase::storage_order_type new_order(gl_order(orig_order));

      if (!(new_order == orig_order))
        {
          // Reorder as interleaved.
          const PixelBufferBase::size_type *shape = v->shape();

          T gl_buf(new typename T::element_type(boost::extents[shape[0]][shape[1]][shape[2]][shape[3]][shape[4]][shape[5]][shape[6]][shape[7]][shape[8]],
                                                v->pixelType(),
                                                v->endianType(),
                                                new_order));
          *gl_buf = *v;
          src_buffer = gl_buf;
        }

      // In interleaved order.
      glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // MultiArray buffers are packed

      glBindTexture(GL_TEXTURE_2D, textureid);
      check_gl("Bind texture");
      glTexSubImage2D(GL_TEXTURE_2D, // target
                      0,  // level, 0 = base, no minimap,
                      0, 0, // x, y
                      tprop.w,  // width
                      tprop.h,  // height
                      tprop.external_format,  // format
                      tprop.external_type, // type
                      //                      testdata);
                      v->data());
      check_gl("Texture set pixels in subregion");
      glGenerateMipmap(GL_TEXTURE_2D);
      check_gl("Generate mipmaps");
    }

    template <typename T>
    typename boost::enable_if_c<
      boost::is_complex<T>::value, void
      >::type
    operator() (const std::shared_ptr<PixelBuffer<T>>& /* v */)
    {
      /// @todo Conversion from complex.
    }

  };

}
#endif

Image2D::Image2D()
  : m_vertices(0)
  , m_image_vertices(0)
  , m_image_texcoords(0)
  , m_image_elements(0)
  , m_num_image_elements(0)
  , m_textureid(0)
  , m_lutid(0)
  , m_texmin(0.0f)
  , m_texmax(0.1f)
  , m_texcorr(1.0f)
{}

Image2D::~Image2D() {}

void
Image2D::create()
{
}

void
Image2D::setSize(const glm::vec2& xlim, const glm::vec2& ylim)
{
  const std::array<GLfloat, 12> square_vertices{
    xlim[0], ylim[0], 0, 
    xlim[1], ylim[0], 0,
    xlim[1], ylim[1], 0,
    xlim[0], ylim[1], 0
  };

  if (m_vertices == 0) {
    glGenVertexArrays(1, &m_vertices);
  }
  glBindVertexArray(m_vertices);

  if (m_image_vertices == 0) {
    glGenBuffers(1, &m_image_vertices);
  }
  glBindBuffer(GL_ARRAY_BUFFER, m_image_vertices);
  glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * square_vertices.size(), square_vertices.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(0);

  glm::vec2 texxlim(0.0, 1.0);
  glm::vec2 texylim(0.0, 1.0);
  std::array<GLfloat, 8> square_texcoords{ texxlim[0], texylim[0], texxlim[1], texylim[0],
                                           texxlim[1], texylim[1], texxlim[0], texylim[1] };

  if (m_image_texcoords == 0) {
    glGenBuffers(1, &m_image_texcoords);
  }
  glBindBuffer(GL_ARRAY_BUFFER, m_image_texcoords);
  glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * square_texcoords.size(), square_texcoords.data(), GL_STATIC_DRAW);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(1);

  std::array<GLushort, 6> square_elements{ // front
                                           0, 1, 2, 2, 3, 0
  };

  if (m_image_elements == 0) {
    glGenBuffers(1, &m_image_elements);
  }
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_image_elements);
  glBufferData(
    GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort) * square_elements.size(), square_elements.data(), GL_STATIC_DRAW);
  m_num_image_elements = square_elements.size();
}

void
Image2D::destroy() 
{
  glDeleteBuffers(1, &m_image_elements);
  glDeleteBuffers(1, &m_image_texcoords);
  glDeleteBuffers(1, &m_image_vertices);
  glDeleteVertexArrays(1, &m_vertices);
}

const glm::vec3&
Image2D::getMin() const
{
  return m_texmin;
}

void
Image2D::setMin(const glm::vec3& min)
{
  m_texmin = min;
}

const glm::vec3&
Image2D::getMax() const
{
  return m_texmax;
}

void
Image2D::setMax(const glm::vec3& max)
{
  m_texmax = max;
}

unsigned int
Image2D::texture()
{
  return m_textureid;
}

unsigned int
Image2D::lut()
{
  return m_lutid;
}
