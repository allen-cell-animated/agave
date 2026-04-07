#pragma once

#include <cstdint>
#include <stddef.h>

struct VolumeDimensions;

namespace FileReaderUtil {

size_t
convertChannelData(uint8_t* dest, const uint8_t* src, const VolumeDimensions& dims);

};
