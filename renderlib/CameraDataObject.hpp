#pragma once

#include "core/prty/prtyFloat.hpp"
#include "core/prty/prtyInt8.hpp"
#include "core/prty/prtyBoolean.hpp"
#include "core/prty/prtyVector3d.hpp"

class CameraDataObject
{
public:
  CameraDataObject() {}

  prtyFloat Exposure{ "Exposure", 0.75f };
  prtyInt8 ExposureIterations{ "ExposureIterations", 1 };
  prtyBoolean NoiseReduction{ "NoiseReduction", false };
  prtyFloat ApertureSize{ "ApertureSize", 0.0f };
  prtyFloat FieldOfView{ "FieldOfView", 30.0f }; // degrees
  prtyFloat FocalDistance{ "FocalDistance", 0.0f };

  prtyVector3d Position{ "Position", glm::vec3(0.0f, 0.0f, 0.0f) };
  prtyVector3d Target{ "Target", glm::vec3(0.0f, 0.0f, -1.0f) };
  prtyFloat NearPlane{ "NearPlane", 0.1f };
  prtyFloat FarPlane{ "FarPlane", 1000.0f };
  prtyFloat Roll{ "Roll", 0.0f }; // tilt angle in degrees
};
