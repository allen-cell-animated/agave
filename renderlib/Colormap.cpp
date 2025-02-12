#include "Colormap.h"

#include "Logging.h"

#include <algorithm>
#include <array>
#include <tuple>
#include <vector>

const std::string ColorRamp::NO_COLORMAP_NAME = "none";

std::vector<uint8_t>
colormapFromControlPoints(std::vector<ColorControlPoint> pts, size_t length)
{
  // pts is piecewise linear from first to last control point.
  // pts is in order of increasing x value (the first element of the pair)
  // pts[0].first === 0
  // pts[pts.size()-1].first === 1
  // Currently all callers satisfy this but we should check.

  if (pts.size() < 2) {
    LOG_ERROR << "Need at least 2 control points to make a colormap";
    return {};
  }
  if (pts[0].first != 0.0f) {
    LOG_ERROR << "colormapFromControlPoints: First control point must be at 0.0";
    return {};
  }
  if (pts[pts.size() - 1].first != 1.0f) {
    LOG_ERROR << "colormapFromControlPoints: Last control point must be at 1.0";
    return {};
  }

  std::vector<uint8_t> lut(length * 4, 0);

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
  // IF THIS GETS MODIFIED THEN OLD FILES USING IT WILL NO LONGER HAVE THE SAME COLORS ON LOAD
  static const std::vector<std::array<uint8_t, 3>> modifiedGlasbey = {
    { 0, 0, 0 },       { 255, 255, 0 },   { 255, 25, 255 },  { 0, 147, 147 },   { 156, 64, 0 },    { 88, 0, 199 },
    { 241, 235, 255 }, { 20, 75, 0 },     { 0, 188, 1 },     { 255, 159, 98 },  { 145, 144, 255 }, { 93, 0, 63 },
    { 0, 255, 214 },   { 255, 0, 95 },    { 120, 100, 119 }, { 0, 73, 96 },     { 140, 136, 74 },  { 82, 207, 255 },
    { 207, 152, 186 }, { 157, 0, 177 },   { 191, 211, 152 }, { 0, 107, 210 },   { 163, 51, 91 },   { 88, 70, 43 },
    { 255, 255, 0 },   { 156, 171, 174 }, { 0, 132, 65 },    { 92, 16, 0 },     { 0, 0, 143 },     { 240, 81, 0 },
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

std::tuple<float, float, float>
hsvToRgb(float h, float s, float v)
{
  float r, g, b;

  int i = static_cast<int>(h * 6);
  float f = (h * 6) - i;
  float p = v * (1 - s);
  float q = v * (1 - f * s);
  float t = v * (1 - (1 - f) * s);

  switch (i % 6) {
    case 0:
      r = v;
      g = t;
      b = p;
      break;
    case 1:
      r = q;
      g = v;
      b = p;
      break;
    case 2:
      r = p;
      g = v;
      b = t;
      break;
    case 3:
      r = p;
      g = q;
      b = v;
      break;
    case 4:
      r = t;
      g = p;
      b = v;
      break;
    case 5:
      r = v;
      g = p;
      b = q;
      break;
  }

  return { r, g, b };
}

uint8_t*
colormapRandomized(size_t length)
{
  uint8_t* lut = new uint8_t[length * 4]{ 0 };

  float r, g, b;
  for (size_t x = 0; x < length; ++x) {
    std::tuple<float, float, float> rgb = hsvToRgb(
      (float)rand() / RAND_MAX, (float)rand() / RAND_MAX * 0.25 + 0.75, (float)rand() / RAND_MAX * 0.75 + 0.25
    );
    r = std::get<0>(rgb);
    g = std::get<1>(rgb);
    b = std::get<2>(rgb);
    lut[x * 4 + 0] = r * 255;
    lut[x * 4 + 1] = g * 255;
    lut[x * 4 + 2] = b * 255;
    lut[x * 4 + 3] = 255;
  }
  return lut;
}

ColorRamp
ColorRamp::createLabels(size_t length)
{
  ColorRamp labels;
  labels.m_name = "Labels";
  uint8_t* lut = modifiedGlasbeyColormap(length);
  // copy lut values into std::vector m_colormap
  labels.m_colormap = std::vector<uint8_t>(lut, lut + length * 4);
  // Labels will have no m_stops.
  // the name "Labels" is a special case.
  return labels;
}

// 11 stops: 0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1
// The names are used for IO and therefore should not be changed.
static const std::vector<ColorRamp> builtInGradients = { ColorRamp(),
                                                         ColorRamp(
                                                           "greyscale",
                                                           stringListToGradient({
                                                             "#000000",
                                                             "#ffffff",
                                                           })
                                                         ),
                                                         ColorRamp(
                                                           "cool",
                                                           stringListToGradient({ "#6e40aa",
                                                                                  "#6154c8",
                                                                                  "#4c6edb",
                                                                                  "#368ce1",
                                                                                  "#24aad8",
                                                                                  "#1ac7c2",
                                                                                  "#1ddea3",
                                                                                  "#30ee83",
                                                                                  "#52f667",
                                                                                  "#7ef658",
                                                                                  "#7ef658" })
                                                         ),
                                                         { "viridis",
                                                           stringListToGradient({ "#440154",
                                                                                  "#482475",
                                                                                  "#414487",
                                                                                  "#355f8d",
                                                                                  "#2a788e",
                                                                                  "#21908d",
                                                                                  "#22a884",
                                                                                  "#42be71",
                                                                                  "#7ad151",
                                                                                  "#bddf26",
                                                                                  "#bddf26" }) },
                                                         { "inferno",
                                                           stringListToGradient({ "#000004",
                                                                                  "#160b39",
                                                                                  "#420a68",
                                                                                  "#6a176e",
                                                                                  "#932667",
                                                                                  "#ba3655",
                                                                                  "#dd513a",
                                                                                  "#f3761b",
                                                                                  "#fca50a",
                                                                                  "#f6d746",
                                                                                  "#f6d746" }) },
                                                         { "magma",
                                                           stringListToGradient({ "#000004",
                                                                                  "#140e36",
                                                                                  "#3b0f70",
                                                                                  "#641a80",
                                                                                  "#8c2981",
                                                                                  "#b5367a",
                                                                                  "#de4968",
                                                                                  "#f66e5c",
                                                                                  "#fe9f6d",
                                                                                  "#fecf92",
                                                                                  "#fecf92" }) },
                                                         { "plasma",
                                                           stringListToGradient({ "#0d0887",
                                                                                  "#41049d",
                                                                                  "#6a00a8",
                                                                                  "#8f0da4",
                                                                                  "#b12a90",
                                                                                  "#cb4679",
                                                                                  "#e16462",
                                                                                  "#f1834c",
                                                                                  "#fca636",
                                                                                  "#fcce25",
                                                                                  "#fcce25" }) },
                                                         { "warm",
                                                           stringListToGradient({ "#6e40aa",
                                                                                  "#963db3",
                                                                                  "#bf3caf",
                                                                                  "#e3419e",
                                                                                  "#fe4b83",
                                                                                  "#ff5e64",
                                                                                  "#ff7747",
                                                                                  "#fb9633",
                                                                                  "#e2b72f",
                                                                                  "#c7d63c",
                                                                                  "#c7d63c" }) },
                                                         { "spectral",
                                                           stringListToGradient({ "#9e0142",
                                                                                  "#d13b4b",
                                                                                  "#f0704a",
                                                                                  "#fcab63",
                                                                                  "#fedc8c",
                                                                                  "#fbf8b0",
                                                                                  "#e0f3a1",
                                                                                  "#aadda2",
                                                                                  "#69bda9",
                                                                                  "#4288b5",
                                                                                  "#4288b5" }) },
                                                         { "rainbow",
                                                           stringListToGradient({ "#6e40aa",
                                                                                  "#be3caf",
                                                                                  "#fe4b83",
                                                                                  "#ff7747",
                                                                                  "#e3b62f",
                                                                                  "#b0ef5a",
                                                                                  "#53f666",
                                                                                  "#1edfa2",
                                                                                  "#23acd8",
                                                                                  "#4c6fdc",
                                                                                  "#4c6fdc" }) },
                                                         { "cubehelix",
                                                           stringListToGradient({ "#000000",
                                                                                  "#1a1530",
                                                                                  "#163d4e",
                                                                                  "#1f6642",
                                                                                  "#53792f",
                                                                                  "#a07949",
                                                                                  "#d07e93",
                                                                                  "#d09cd9",
                                                                                  "#c1caf3",
                                                                                  "#d2eeef",
                                                                                  "#d2eeef" }) },
                                                         { "RdYlGn",
                                                           stringListToGradient({ "#a50026",
                                                                                  "#d3322b",
                                                                                  "#f16d43",
                                                                                  "#fcab63",
                                                                                  "#fedc8c",
                                                                                  "#f9f7ae",
                                                                                  "#d7ee8e",
                                                                                  "#a4d86f",
                                                                                  "#64bc61",
                                                                                  "#23964f",
                                                                                  "#23964f" }) },
                                                         { "RdYlBu",
                                                           stringListToGradient({ "#a50026",
                                                                                  "#d3322b",
                                                                                  "#f16d43",
                                                                                  "#fcab64",
                                                                                  "#fedc90",
                                                                                  "#faf8c0",
                                                                                  "#dcf1ec",
                                                                                  "#abd6e8",
                                                                                  "#76abd0",
                                                                                  "#4a74b4",
                                                                                  "#4a74b4" }) },
                                                         { "PuBuGn",
                                                           stringListToGradient({ "#fff7fb",
                                                                                  "#efe7f2",
                                                                                  "#dbd8ea",
                                                                                  "#bfc9e2",
                                                                                  "#98b9d9",
                                                                                  "#6aa8cf",
                                                                                  "#4096c0",
                                                                                  "#1987a0",
                                                                                  "#047877",
                                                                                  "#016353",
                                                                                  "#016353" }) },
                                                         { "YlGnBu",
                                                           stringListToGradient({ "#ffffd9",
                                                                                  "#eff9bd",
                                                                                  "#d5efb3",
                                                                                  "#a9ddb7",
                                                                                  "#74c9bd",
                                                                                  "#46b4c2",
                                                                                  "#2897bf",
                                                                                  "#2073b2",
                                                                                  "#234ea0",
                                                                                  "#1d3185",
                                                                                  "#1d3185" }) },
                                                         { "GnBu",
                                                           stringListToGradient({ "#f7fcf0",
                                                                                  "#e5f5df",
                                                                                  "#d4eece",
                                                                                  "#bde5bf",
                                                                                  "#9fd9bb",
                                                                                  "#7bcbc4",
                                                                                  "#58b7cd",
                                                                                  "#399cc6",
                                                                                  "#1e7eb7",
                                                                                  "#0b60a1",
                                                                                  "#0b60a1" }) },
                                                         { "YlOrRd",
                                                           stringListToGradient({ "#ffffcc",
                                                                                  "#fff1a9",
                                                                                  "#fee087",
                                                                                  "#fec966",
                                                                                  "#feab4b",
                                                                                  "#fd893c",
                                                                                  "#fa5c2e",
                                                                                  "#ec3023",
                                                                                  "#d31121",
                                                                                  "#af0225",
                                                                                  "#af0225" }) },
                                                         { "YlOrBr",
                                                           stringListToGradient({ "#ffffe5",
                                                                                  "#fff8c4",
                                                                                  "#feeba2",
                                                                                  "#fed676",
                                                                                  "#febb4a",
                                                                                  "#fb9a2c",
                                                                                  "#ee7919",
                                                                                  "#d85b0a",
                                                                                  "#b74304",
                                                                                  "#8f3204",
                                                                                  "#8f3204" }) },
                                                         { "RdBu",
                                                           stringListToGradient({ "#67001f",
                                                                                  "#ab202e",
                                                                                  "#d55f50",
                                                                                  "#f0a285",
                                                                                  "#fad6c3",
                                                                                  "#f2efee",
                                                                                  "#cde3ee",
                                                                                  "#90c2dd",
                                                                                  "#4b94c4",
                                                                                  "#2265a3",
                                                                                  "#2265a3" }) },
                                                         { "BuGn",
                                                           stringListToGradient({ "#f7fcfd",
                                                                                  "#e8f6f9",
                                                                                  "#d5efed",
                                                                                  "#b7e4da",
                                                                                  "#8fd4c1",
                                                                                  "#69c2a3",
                                                                                  "#49b17f",
                                                                                  "#2f995a",
                                                                                  "#157f3c",
                                                                                  "#036429",
                                                                                  "#036429" }) },
                                                         { "BuPu",
                                                           stringListToGradient({ "#f7fcfd",
                                                                                  "#e4eff5",
                                                                                  "#ccddec",
                                                                                  "#b2cae1",
                                                                                  "#9cb3d5",
                                                                                  "#8f95c6",
                                                                                  "#8c74b5",
                                                                                  "#8952a5",
                                                                                  "#852d8f",
                                                                                  "#730f71",
                                                                                  "#730f71" }) },
                                                         { "PuBu",
                                                           stringListToGradient({ "#fff7fb",
                                                                                  "#efeaf4",
                                                                                  "#dbdaeb",
                                                                                  "#bfc9e2",
                                                                                  "#9cb9d9",
                                                                                  "#72a8cf",
                                                                                  "#4494c3",
                                                                                  "#1b7db6",
                                                                                  "#0668a1",
                                                                                  "#045281",
                                                                                  "#045281" }) },
                                                         { "RdPu",
                                                           stringListToGradient({ "#fff7f3",
                                                                                  "#fde4e1",
                                                                                  "#fccfcc",
                                                                                  "#fbb5bc",
                                                                                  "#f993b0",
                                                                                  "#f369a3",
                                                                                  "#e03f98",
                                                                                  "#c11889",
                                                                                  "#99037c",
                                                                                  "#710174",
                                                                                  "#710174" }) },
                                                         { "PuRd",
                                                           stringListToGradient({ "#f7f4f9",
                                                                                  "#eae3f0",
                                                                                  "#dcc9e2",
                                                                                  "#d0aad2",
                                                                                  "#d08ac2",
                                                                                  "#dd63ae",
                                                                                  "#e33890",
                                                                                  "#d71c6c",
                                                                                  "#b80b50",
                                                                                  "#8f023a",
                                                                                  "#8f023a" }) },
                                                         { "YlGn",
                                                           stringListToGradient({ "#ffffe5",
                                                                                  "#f7fcc4",
                                                                                  "#e4f4ac",
                                                                                  "#c7e89b",
                                                                                  "#a2d88a",
                                                                                  "#78c578",
                                                                                  "#4eaf63",
                                                                                  "#2f944e",
                                                                                  "#15793f",
                                                                                  "#036034",
                                                                                  "#036034" }) },
                                                         { "OrRd",
                                                           stringListToGradient({ "#fff7ec",
                                                                                  "#feebcf",
                                                                                  "#fddcaf",
                                                                                  "#fdca94",
                                                                                  "#fdb07a",
                                                                                  "#fa8e5d",
                                                                                  "#f16c49",
                                                                                  "#e04630",
                                                                                  "#c81e13",
                                                                                  "#a70403",
                                                                                  "#a70403" }) },
                                                         { "PiYG",
                                                           stringListToGradient({ "#8e0152",
                                                                                  "#c0267e",
                                                                                  "#dd72ad",
                                                                                  "#f0b2d6",
                                                                                  "#fadded",
                                                                                  "#f5f3ef",
                                                                                  "#e1f2ca",
                                                                                  "#b7de88",
                                                                                  "#81bb48",
                                                                                  "#4f9125",
                                                                                  "#4f9125" }) },
                                                         { "PRGn",
                                                           stringListToGradient({ "#40004b",
                                                                                  "#722e80",
                                                                                  "#9a6daa",
                                                                                  "#c1a4cd",
                                                                                  "#e3d2e6",
                                                                                  "#eff0ef",
                                                                                  "#d6eed1",
                                                                                  "#a2d79f",
                                                                                  "#5dad65",
                                                                                  "#217939",
                                                                                  "#217939" }) },
                                                         { "PuOr",
                                                           stringListToGradient({ "#7f3b08",
                                                                                  "#b15a09",
                                                                                  "#dd841f",
                                                                                  "#f8b664",
                                                                                  "#fdddb2",
                                                                                  "#f3eeea",
                                                                                  "#d7d7e9",
                                                                                  "#b0aad0",
                                                                                  "#8170ad",
                                                                                  "#552d84",
                                                                                  "#552d84" }) },
                                                         { "RdGy",
                                                           stringListToGradient({ "#67001f",
                                                                                  "#ab202e",
                                                                                  "#d55f50",
                                                                                  "#f0a285",
                                                                                  "#fcd8c4",
                                                                                  "#faf4f0",
                                                                                  "#dfdfdf",
                                                                                  "#b8b8b8",
                                                                                  "#868686",
                                                                                  "#4e4e4e",
                                                                                  "#4e4e4e" }) },
                                                         { "BrBG",
                                                           stringListToGradient({ "#543005",
                                                                                  "#8b530f",
                                                                                  "#bc8434",
                                                                                  "#ddbd7b",
                                                                                  "#f2e4bf",
                                                                                  "#eef1ea",
                                                                                  "#c3e7e2",
                                                                                  "#80c9bf",
                                                                                  "#399890",
                                                                                  "#0a675f",
                                                                                  "#0a675f" }) },
                                                         ColorRamp::createLabels() };

const std::vector<ColorRamp>&
getBuiltInGradients()
{
  return builtInGradients;
}

// "none" and "Labels" are special cases.
const ColorRamp&
ColorRamp::colormapFromName(const std::string& name)
{
  for (auto& gspec : getBuiltInGradients()) {
    // the name is actually in our list:
    if (gspec.m_name == name) {
      return gspec;
    }
  }

  LOG_ERROR << "Unknown colormap name: " << name << ". Falling back to no colormap.";
  // use "none" as a special case when the map is not found.
  return colormapFromName(NO_COLORMAP_NAME);
}

void
ColorRamp::createColormap(size_t length)
{
  m_colormap = colormapFromControlPoints(m_stops, length);
}

void
ColorRamp::debugPrintColormap() const
{
  // stringify for output
  std::stringstream ss;
  for (size_t x = 0; x < 256 * 4; ++x) {
    ss << (int)m_colormap[x] << ", ";
  }
  LOG_DEBUG << "COLORMAP: " << ss.str();
}

ColorRamp::ColorRamp()
{
  m_name = NO_COLORMAP_NAME;
  m_stops = stringListToGradient({ "#ffffff", "#ffffff" });
  createColormap();
}
