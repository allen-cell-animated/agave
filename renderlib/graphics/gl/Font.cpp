#include "Font.h"

#include "Util.h"

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb/stb_truetype.h"

#include <stdio.h>

static constexpr int s_textureSize = 512;

GLuint
generateFontTexture(unsigned char* temp_bitmap)
{
  // expand temp_bitmap to RGBA
  unsigned char* expanded_bitmap = new unsigned char[s_textureSize * s_textureSize * 4];
  for (int i = 0; i < s_textureSize * s_textureSize; ++i) {
    expanded_bitmap[i * 4 + 0] = 255;
    expanded_bitmap[i * 4 + 1] = 255;
    expanded_bitmap[i * 4 + 2] = 255;
    expanded_bitmap[i * 4 + 3] = temp_bitmap[i];
  }

  GLuint ftex = 0;
  glGenTextures(1, &ftex);
  glBindTexture(GL_TEXTURE_2D, ftex);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, s_textureSize, s_textureSize, 0, GL_RGBA, GL_UNSIGNED_BYTE, expanded_bitmap);
  // glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA, s_textureSize, s_textureSize, 0, GL_ALPHA, GL_UNSIGNED_BYTE, temp_bitmap);
  check_gl("load font texture");
  // can free temp_bitmap at this point
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  check_gl("set filtering on font texture");
  delete[] expanded_bitmap;
  return ftex;
}

Font::Font()
{
  m_texID = 0;
}

Font::~Font()
{
  unload();
}

void
Font::load(const char* filename)
{
  // lazy init? // TODO fix this to do explicit init?
  if (m_texID != 0) {
    return;
  }

  FILE* file = fopen(filename, "rb");
  if (!file) {
    return;
  }

  fseek(file, 0, SEEK_END);
  long size = ftell(file);
  fseek(file, 0, SEEK_SET);

  unsigned char* buffer = (unsigned char*)malloc(size);
  fread(buffer, 1, size, file);
  fclose(file);

  unsigned char temp_bitmap[s_textureSize * s_textureSize];
  stbtt_BakeFontBitmap(buffer,
                       0,
                       32.0,
                       temp_bitmap,
                       s_textureSize,
                       s_textureSize,
                       m_firstChar,
                       m_numChars,
                       m_cdata); // no guarantee this fits!

  free(buffer);

  m_texID = generateFontTexture(temp_bitmap);
}

void
Font::unload()
{
  glDeleteTextures(1, &m_texID);
  m_texID = 0;
}

// void
// Font::drawText(const char* text, float x, float y, float scale, float color[4])
// {
//   float xpos = x;
//   float ypos = y;

//   // while (*text) {
//   //   int advance, lsb, x0, y0, x1, y1;
//   //   stbtt_GetCodepointHMetrics(&m_font, *text, &advance, &lsb);
//   //   stbtt_GetCodepointBitmapBox(&m_font, *text, scale, scale, &x0, &y0, &x1, &y1);
//   //   stbtt_MakeCodepointBitmap(&m_font, m_bitmap, x1 - x0, y1 - y0, m_bitmapWidth, scale, scale, *text);
//   //   drawBitmap(m_bitmap, xpos + x0, ypos + y0, x1 - x0, y1 - y0, color);
//   //   xpos += (advance * scale);
//   //   text++;
//   // }

//   // assume orthographic projection with units = screen pixels, origin at top left
//   glEnable(GL_BLEND);
//   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
//   glEnable(GL_TEXTURE_2D);
//   glBindTexture(GL_TEXTURE_2D, ftex);
//   glBegin(GL_QUADS);
//   while (*text) {
//     if (*text >= m_firstChar && *text < (m_firstChar + m_numChars)) {
//       stbtt_aligned_quad q;
//       stbtt_GetBakedQuad(cdata, 512, 512, *text - m_firstChar, &x, &y, &q, 1); // 1=opengl & d3d10+,0=d3d9
//       glTexCoord2f(q.s0, q.t0);
//       glVertex2f(q.x0, q.y0);
//       glTexCoord2f(q.s1, q.t0);
//       glVertex2f(q.x1, q.y0);
//       glTexCoord2f(q.s1, q.t1);
//       glVertex2f(q.x1, q.y1);
//       glTexCoord2f(q.s0, q.t1);
//       glVertex2f(q.x0, q.y1);
//     }
//     ++text;
//   }
//   glEnd();
// }

bool
Font::getBakedQuad(char char_index, float* x, float* y, stbtt_aligned_quad* q)
{
  if (char_index >= m_firstChar && char_index < (m_firstChar + m_numChars)) {
    stbtt_GetBakedQuad(
      m_cdata, s_textureSize, s_textureSize, char_index - m_firstChar, x, y, q, 1); // 1=opengl & d3d10+,0=d3d9
    // because our coordinate system has 0,0 at bottom left, we need to fixup these y coordinates
    size_t offset = char_index - m_firstChar;
    const stbtt_bakedchar* b = m_cdata + offset;

    float height = q->y1 - q->y0;
    float h0 = q->y0;
    // this is our base line?
    float h1 = q->y1;
    q->y0 = h1 - 2.0 * b->yoff - height;
    q->y1 = h0 - 2.0 * b->yoff - height;
    return true;
  }
  return false;
}

float
Font::getStringWidth(std::string stext)
{
  const char* text = stext.c_str();
  float xpos = 0;
  float ypos = 0;
  float width = 0;
  while (*text) {
    if (*text >= m_firstChar && *text < (m_firstChar + m_numChars)) {
      // stbtt_aligned_quad q;
      // stbtt_GetBakedQuad(m_cdata, s_textureSize, s_textureSize, *text - m_firstChar, &xpos, &ypos, &q, 1);
      // width += q.x1 - q.x0;
      // the above may be a bit more realistic but slightly more expensive to compute?
      size_t offset = *text - m_firstChar;
      const stbtt_bakedchar* b = m_cdata + offset;
      width += b->xadvance;
    }
    ++text;
  }
  return width;
}

float
Font::getStringHeight(std::string stext)
{
  const char* text = stext.c_str();
  float xpos = 0;
  float ypos = 0;
  float height = 0;
  while (*text) {
    if (*text >= m_firstChar && *text < (m_firstChar + m_numChars)) {
      // stbtt_aligned_quad q;
      // stbtt_GetBakedQuad(m_cdata, s_textureSize, s_textureSize, *text - m_firstChar, &xpos, &ypos, &q, 1);
      // height = max(height, abs(q.y1 - q.y0));
      // the above may be a bit more realistic but slightly more expensive to compute?
      size_t offset = *text - m_firstChar;
      const stbtt_bakedchar* b = m_cdata + offset;
      height = std::max(height, std::abs((float)b->y1 - (float)b->y0));
    }
    ++text;
  }
  return height;
}