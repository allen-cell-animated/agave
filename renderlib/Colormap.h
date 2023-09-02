#pragma once

#include <inttypes.h>

struct ColorControlPoint
{
  float first;
  uint8_t r, g, b, a;
  ColorControlPoint(float x, uint8_t r, uint8_t g, uint8_t b, uint8_t a)
    : first(x)
    , r(r)
    , g(g)
    , b(b)
    , a(a)
  {
  }
  ColorControlPoint(float x, float r, float g, float b, float a)
    : first(x)
    , r(r * 255.0f)
    , g(g * 255.0f)
    , b(b * 255.0f)
    , a(a * 255.0f)
  {
  }
};

uint8_t*
colormapFromControlPoints(std::vector<ColorControlPoint> pts, size_t length = 256);
