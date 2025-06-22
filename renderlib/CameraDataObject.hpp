#pragma once

#include "core/prty/prtyFloat.hpp"
#include "core/prty/prtyInt8.hpp"
#include "core/prty/prtyBoolean.hpp"

class CameraDataObject
{
public:
  CameraDataObject() {}

  prtyFloat Exposure{ "Exposure", 0.75f };
  prtyInt8 ExposureIterations{ "ExposureIterations", 1 };
  prtyBoolean NoiseReduction{ "NoiseReduction", false };
  prtyFloat ApertureSize{ "ApertureSize", 0.0f };
  prtyFloat FieldOfView{ "FieldOfView", 30.0f };
  prtyFloat FocalDistance{ "FocalDistance", 0.0f };
};
