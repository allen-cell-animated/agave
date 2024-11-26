#pragma once

#include "graphics/gl/Util.h"

class GLThickLinesShader : public GLShaderProgram
{
public:
  GLThickLinesShader();

  ~GLThickLinesShader() {}

  void configure(bool display, GLuint textureId);
  void cleanup();

  int m_loc_proj;
  int m_loc_vpos;
  int m_loc_vuv;
  int m_loc_vcol;
  int m_loc_vcode;
  int m_loc_thickness;
  int m_loc_resolution;
};
