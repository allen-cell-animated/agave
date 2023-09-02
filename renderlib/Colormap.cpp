#include "Colormap.h"

uint8_t*
colormapFromControlPoints(std::vector<ColorControlPoint> pts, size_t length)
{
  // pts is piecewise linear from first to last control point.
  // pts is in order of increasing x value (the first element of the pair)
  // pts[0].first === 0
  // pts[pts.size()-1].first === 1

  uint8_t* lut = new uint8_t[length * 4]{ 0 };

  for (size_t x = 0; x < length; ++x) {
    float fx = (float)x / (float)(length - 1);
    // find the interval of control points that contains fx.
    for (size_t i = 0; i < pts.size() - 1; ++i) {
      // am i in between?
      if ((fx >= pts[i].first) && (fx <= pts[i + 1].first)) {
        // what fraction of this interval in x?
        float fxi = (fx - pts[i].first) / (pts[i + 1].first - pts[i].first);
        // use that fraction against y range
        // TODO test rounding error
        lut[x * 4 + 0] = pts[i].r + fxi * (pts[i + 1].r - pts[i].r);
        lut[x * 4 + 1] = pts[i].g + fxi * (pts[i + 1].g - pts[i].g);
        lut[x * 4 + 2] = pts[i].b + fxi * (pts[i + 1].b - pts[i].b);
        lut[x * 4 + 3] = pts[i].a + fxi * (pts[i + 1].a - pts[i].a);
        break;
      }
    }
  }
  return lut;
}
