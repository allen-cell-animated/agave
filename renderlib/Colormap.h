#pragma once

#include <inttypes.h>
#include <stddef.h>
#include <string>
#include <vector>

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
  ColorControlPoint(float x, unsigned int r, unsigned int g, unsigned int b, unsigned int a)
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

  ColorControlPoint(float t, const std::string& hexcolor)
    : first(t)
    , r(0)
    , g(0)
    , b(0)
    , a(0)
  {
    if (hexcolor.size() == 7) {
      unsigned int ru = 0, gu = 0, bu = 0;
      sscanf(hexcolor.c_str(), "#%02x%02x%02x", &ru, &gu, &bu);
      r = ru;
      g = gu;
      b = bu;
      a = 255;
    }
  }
  ColorControlPoint(float t, uint32_t rgb)
  {
    first = t;
    a = 255;
    r = (rgb >> 16) & 0xff;
    g = (rgb >> 8) & 0xff;
    b = (rgb >> 0) & 0xff;
  }
};

uint8_t*
colormapFromControlPoints(std::vector<ColorControlPoint> pts, size_t length = 256);
uint8_t*
modifiedGlasbeyColormap(size_t length = 256);
uint8_t*
colormapRandomized(size_t length = 256);
uint8_t*
colormapFromColormap(uint8_t* colormap, size_t length = 256);

std::vector<ColorControlPoint>
stringListToGradient(const std::vector<std::string>& colors);
