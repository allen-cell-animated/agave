#include "Colormap.h"

#include <QColor>

#include <vector>
#include <array>

uint8_t*
colormapFromColormap(uint8_t* colormap, size_t length)
{
  // basically just copy the whole thing.
  uint8_t* lut = new uint8_t[length * 4]{ 0 };
  for (size_t x = 0; x < length; ++x) {
    lut[x * 4 + 0] = colormap[x * 4 + 0];
    lut[x * 4 + 1] = colormap[x * 4 + 1];
    lut[x * 4 + 2] = colormap[x * 4 + 2];
    lut[x * 4 + 3] = colormap[x * 4 + 3];
  }
  return lut;
}

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

std::vector<ColorControlPoint>
stringListToGradient(const std::vector<std::string>& colors)
{
  std::vector<ColorControlPoint> stops;
  size_t n = colors.size();
  for (int i = 0; i < n; ++i) {
    stops.push_back(ColorControlPoint(i / (n - 1.0f), colors[i]));
  }
  return stops;
}

uint8_t*
modifiedGlasbeyColormap(size_t length)
{
  static std::vector<std::array<uint8_t, 3>> modifiedGlasbey = {
    { 0, 0, 0 },       { 255, 255, 0 },   { 255, 25, 255 },  { 0, 147, 147 },   { 156, 64, 0 },    { 88, 0, 199 },
    { 241, 235, 255 }, { 20, 75, 0 },     { 0, 188, 1 },     { 255, 159, 98 },  { 145, 144, 255 }, { 93, 0, 63 },
    { 0, 255, 214 },   { 255, 0, 95 },    { 120, 100, 119 }, { 0, 73, 96 },     { 140, 136, 74 },  { 82, 207, 255 },
    { 207, 152, 186 }, { 157, 0, 177 },   { 191, 211, 152 }, { 0, 107, 210 },   { 163, 51, 91 },   { 88, 70, 43 },
    { 255, 255, 0 },   { 156, 171, 174 }, { 0, 132, 65 },    { 92, 16, 0 },     { 0, 0, 143 },      { 240, 81, 0 },
    { 205, 170, 0 },   { 182, 114, 100 }, { 76, 190, 141 },  { 148, 60, 255 },  { 82, 54, 100 },   { 73, 101, 93 },
    { 110, 132, 165 }, { 175, 105, 192 }, { 208, 184, 255 }, { 255, 211, 190 }, { 212, 255, 237 }, { 255, 123, 137 },
    { 96, 98, 0 },     { 222, 0, 157 },   { 0, 159, 249 },   { 197, 120, 1 },   { 0, 1, 255 },     { 197, 1, 29 },
    { 190, 163, 136 }, { 98, 90, 157 },   { 255, 144, 255 }, { 160, 205, 0 },   { 255, 215, 97 },  { 107, 59, 73 },
    { 101, 144, 0 },   { 124, 131, 125 }, { 255, 255, 195 }, { 149, 215, 214 }, { 18, 112, 141 },  { 255, 195, 239 },
    { 195, 102, 146 }, { 140, 0, 30 },    { 138, 177, 93 },  { 135, 98, 59 },   { 183, 209, 245 }, { 163, 153, 193 },
    { 16, 189, 193 },  { 255, 102, 194 }, { 48, 57, 118 },   { 77, 82, 99 },    { 205, 192, 201 }, { 94, 63, 255 },
    { 197, 135, 255 }, { 195, 0, 255 },   { 0, 80, 57 },     { 139, 3, 110 },   { 208, 252, 136 }, { 127, 229, 159 },
    { 151, 78, 136 },  { 110, 0, 159 },   { 130, 170, 209 }, { 100, 150, 109 }, { 158, 132, 145 }, { 204, 76, 87 },
    { 66, 0, 124 },    { 255, 172, 180 }, { 136, 119, 190 }, { 144, 86, 89 },   { 109, 52, 0 },    { 93, 116, 74 },
    { 0, 246, 255 },   { 96, 116, 255 },  { 84, 0, 98 },     { 0, 169, 83 },    { 137, 79, 175 },  { 219, 182, 107 },
    { 197, 213, 203 }, { 16, 144, 184 },  { 230, 121, 85 },  { 65, 85, 45 },    { 42, 106, 0 },    { 104, 96, 87 },
    { 255, 165, 4 },   { 2, 220, 95 },    { 151, 183, 151 }, { 147, 109, 0 },   { 247, 0, 29 },    { 195, 50, 190 },
    { 2, 85, 148 },    { 192, 93, 39 },   { 0, 125, 102 },   { 156, 149, 0 },   { 248, 125, 0 },   { 255, 252, 243 },
    { 105, 165, 154 }, { 184, 245, 0 },   { 132, 56, 45 },   { 214, 144, 141 }, { 209, 0, 98 },    { 197, 241, 255 },
    { 222, 214, 3 },   { 163, 180, 255 }, { 90, 124, 130 },  { 105, 26, 42 },   { 186, 150, 73 },  { 114, 79, 0 },
    { 0, 216, 189 },   { 120, 53, 126 },  { 157, 133, 109 }, { 215, 124, 206 }, { 254, 85, 81 },   { 0, 96, 100 },
    { 238, 85, 137 },  { 23, 176, 218 },  { 187, 255, 192 }, { 126, 0, 220 },   { 255, 150, 208 }, { 73, 65, 0 },
    { 216, 90, 255 },  { 176, 33, 135 },  { 163, 110, 255 }, { 64, 71, 177 },   { 49, 0, 186 },    { 186, 196, 86 },
    { 14, 103, 55 },   { 85, 105, 136 },  { 137, 0, 68 },    { 0, 158, 125 },   { 125, 174, 0 },   { 209, 196, 168 },
    { 140, 143, 156 }, { 158, 224, 110 }, { 86, 73, 78 },    { 154, 255, 245 }, { 176, 162, 168 }, { 171, 62, 51 },
    { 103, 153, 169 }, { 146, 116, 156 }, { 106, 80, 124 },  { 77, 131, 204 },  { 179, 182, 207 }, { 160, 25, 0 },
    { 143, 154, 123 }, { 170, 117, 64 },  { 91, 59, 145 },   { 91, 227, 0 },    { 205, 156, 223 }, { 235, 177, 149 },
    { 0, 140, 1 },     { 204, 57, 2 },    { 239, 218, 157 }, { 175, 176, 117 }, { 138, 111, 109 }, { 156, 104, 129 },
    { 97, 42, 85 },    { 167, 229, 200 }, { 129, 186, 199 }, { 254, 219, 232 }, { 120, 123, 19 },  { 99, 104, 111 },
    { 101, 94, 51 },   { 217, 85, 183 },  { 200, 139, 98 },  { 115, 97, 214 },  { 73, 80, 70 },    { 227, 126, 166 },
    { 222, 203, 238 }, { 132, 57, 100 },  { 213, 110, 121 }, { 158, 3, 224 },   { 175, 0, 58 },    { 117, 76, 61 },
    { 89, 127, 109 },  { 139, 229, 255 }, { 125, 33, 0 },    { 123, 117, 89 },  { 133, 147, 204 }, { 179, 134, 184 },
    { 116, 164, 254 }, { 126, 193, 175 }, { 162, 91, 3 },    { 26, 70, 255 },   { 255, 0, 202 },   { 215, 236, 199 },
    { 255, 248, 115 }, { 115, 108, 145 }, { 0, 255, 150 },   { 114, 201, 120 }, { 80, 124, 40 },   { 160, 88, 60 },
    { 68, 81, 128 }
  };
  uint8_t* lut = new uint8_t[length * 4]{ 0 };
  for (size_t x = 0; x < length; ++x) {
    lut[x * 4 + 0] = modifiedGlasbey[x % modifiedGlasbey.size()][0];
    lut[x * 4 + 1] = modifiedGlasbey[x % modifiedGlasbey.size()][1];
    lut[x * 4 + 2] = modifiedGlasbey[x % modifiedGlasbey.size()][2];
    lut[x * 4 + 3] = 255;
  }
  return lut;
}

uint8_t*
colormapRandomized(size_t length)
{
  uint8_t* lut = new uint8_t[length * 4]{ 0 };

  float r, g, b;
  for (size_t x = 0; x < length; ++x) {
    QColor color = QColor::fromHsvF(
      (float)rand() / RAND_MAX, (float)rand() / RAND_MAX * 0.25 + 0.75, (float)rand() / RAND_MAX * 0.75 + 0.25);
    color.getRgbF(&r, &g, &b);
    lut[x * 4 + 0] = r * 255;
    lut[x * 4 + 1] = g * 255;
    lut[x * 4 + 2] = b * 255;
    lut[x * 4 + 3] = 255;
  }
  return lut;
}
