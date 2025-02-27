#include "Fuse.h"

#include "GradientData.h"
#include "ImageXYZC.h"

#include "threading.h"

// fuse: fill volume of color data, plus volume of gradients
// n channels with n colors: use "max" or "avg"
// n channels with gradients: use "max" or "avg"
void
Fuse::fuse(const ImageXYZC* img,
           const std::vector<glm::vec3>& colorsPerChannel,
           const GradientData* channelGradientData,
           const float* channelIsLabels,
           uint8_t** outRGBVolume,
           uint16_t** outGradientVolume)
{
  // todo: this can easily be a cuda kernel that loops over channels and does a max operation, if it has the full volume
  // data in gpu mem.

  // create and zero
  uint8_t* rgbVolume = *outRGBVolume;
  memset(*outRGBVolume, 0, 3 * img->sizeX() * img->sizeY() * img->sizeZ() * sizeof(uint8_t));

  const bool FUSE_THREADED = true;

  parallel_for(
    img->sizeX() * img->sizeY() * img->sizeZ(),
    [&img, &colorsPerChannel, &channelGradientData, &channelIsLabels, &rgbVolume](size_t s, size_t e) {
      float value = 0;
      uint16_t rawvalue = 0;
      float normalizedvalue = 0;
      float lutnormalizedvalue = 0;
      float r = 0, g = 0, b = 0;
      float cr = 0, cg = 0, cb = 0;
      uint8_t ar = 0, ag = 0, ab = 0;
      bool isLabels;

      size_t ncolors = colorsPerChannel.size();
      size_t nch = std::min((size_t)img->sizeC(), ncolors);

      for (uint32_t i = 0; i < nch; ++i) {
        glm::vec3 c = colorsPerChannel[i];
        if (c == glm::vec3(0, 0, 0)) {
          continue;
        }
        r = c.x; // 0..1
        g = c.y;
        b = c.z;
        uint16_t* channeldata = reinterpret_cast<uint16_t*>(img->ptr(i));

        // array of 256 floats
        float* lut = img->channel(i)->m_lut;
        float chmax = (float)img->channel(i)->m_max;
        float chmin = (float)img->channel(i)->m_min;
        // lut = luts[idx][c.enhancement];

        isLabels = channelIsLabels ? (channelIsLabels[i] > 0 ? true : false) : false;
        uint8_t* colormap = img->channel(i)->m_colormap;
        //  get a min/max from the gradient data if possible
        uint16_t imin16 = 0;
        uint16_t imax16 = 0;
        bool hasMinMax = channelGradientData[i].getMinMax(img->channel(i)->m_histogram, &imin16, &imax16);
        uint16_t lutmin = hasMinMax ? imin16 : chmin;
        uint16_t lutmax = hasMinMax ? imax16 : chmax;

        // channel data cx is scalar so loop from s to e
        // fused data is RGB so offset in multiples of 3
        for (size_t cx = s, fx = s * 3; cx < e; cx++, fx += 3) {
          rawvalue = channeldata[cx];
          normalizedvalue = (float)(rawvalue - chmin) / (float)(chmax - chmin);
          value = lut[(int)(normalizedvalue * 255.0 + 0.5)]; // 0..255

          // apply colormap
          // if not labels then do lookup with normalized value
          if (isLabels) {
            cr = colormap ? r * (float)colormap[(rawvalue % 256) * 4 + 0] / 255.0f : r;
            cg = colormap ? g * (float)colormap[(rawvalue % 256) * 4 + 1] / 255.0f : g;
            cb = colormap ? b * (float)colormap[(rawvalue % 256) * 4 + 2] / 255.0f : b;
          } else {
            lutnormalizedvalue = (float)(rawvalue - lutmin) / (float)(lutmax - lutmin);
            if (lutnormalizedvalue < 0.0f) {
              lutnormalizedvalue = 0.0f;
            }
            if (lutnormalizedvalue > 1.0f) {
              lutnormalizedvalue = 1.0f;
            }
            cr = colormap ? r * (float)colormap[(int)(lutnormalizedvalue * 255.0) * 4 + 0] / 255.0f : r;
            cg = colormap ? g * (float)colormap[(int)(lutnormalizedvalue * 255.0) * 4 + 1] / 255.0f : g;
            cb = colormap ? b * (float)colormap[(int)(lutnormalizedvalue * 255.0) * 4 + 2] / 255.0f : b;
          }
          //  what if rgb*value > 1?
          ar = rgbVolume[fx + 0];
          rgbVolume[fx + 0] = std::max(ar, static_cast<uint8_t>(cr * value * 255));
          ag = rgbVolume[fx + 1];
          rgbVolume[fx + 1] = std::max(ag, static_cast<uint8_t>(cg * value * 255));
          ab = rgbVolume[fx + 2];
          rgbVolume[fx + 2] = std::max(ab, static_cast<uint8_t>(cb * value * 255));
        }
      }
    },
    FUSE_THREADED);
}
