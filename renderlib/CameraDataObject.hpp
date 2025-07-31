#pragma once

#include "core/prty/prtyFloat.hpp"
#include "core/prty/prtyEnum.hpp"
#include "core/prty/prtyBoolean.hpp"
#include "core/prty/prtyVector3d.hpp"

class CameraDataObject
{
public:
  CameraDataObject()
  {
    ExposureIterations.SetEnumTag(0, "1");
    ExposureIterations.SetEnumTag(1, "2");
    ExposureIterations.SetEnumTag(2, "4");
    ExposureIterations.SetEnumTag(3, "8");
  }

  prtyFloat Exposure{ "Exposure", 0.75f };
  prtyEnum ExposureIterations{ "ExposureIterations", 0 };
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
