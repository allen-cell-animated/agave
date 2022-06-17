#pragma once

#include "glm.h"

#include <vector>

class ImageXYZC;

// Runs a processing step that applies a color into each channel,
// and then combines the channels to result in a single RGB colored volume
class Fuse
{
public:
  // if channel color is 0, then channel will not contribute.
  // allocates memory for outRGBVolume and outGradientVolume
  static void fuse(const ImageXYZC* img,
                   const std::vector<glm::vec3>& colorsPerChannel,
                   uint8_t** outRGBVolume,
                   uint16_t** outGradientVolume);
};
