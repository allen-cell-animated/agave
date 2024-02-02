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
      // if this ever proves inaccurate, consider using stbtt_GetBakedQuad for height/width info
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
      // if this ever proves inaccurate, consider using stbtt_GetBakedQuad for height/width info
      size_t offset = *text - m_firstChar;
      const stbtt_bakedchar* b = m_cdata + offset;
      height = std::max(height, std::abs((float)b->y1 - (float)b->y0));
    }
    ++text;
  }
  return height;
}
