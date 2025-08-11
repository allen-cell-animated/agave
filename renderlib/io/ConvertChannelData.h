#pragma once

#include <cstdint>
#include <stddef.h>

class VolumeDimensions;

namespace FileReaderUtil {

extern size_t
convertChannelData(uint8_t* dest, const uint8_t* src, const VolumeDimensions& dims);

}