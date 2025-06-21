#pragma once

#include "core/prty/prtyFloat.hpp"
#include "core/prty/prtyInt8.hpp"
#include "core/prty/prtyBoolean.hpp"
#include "CCamera.h"

class CameraDataObject
{
public:
  CameraDataObject()
    : m_camera(nullptr)
  {
    // updatePropsFromCamera();
  }
  CameraDataObject(CCamera* camera);

  prtyFloat Exposure{ "Exposure", 0.75f };
  prtyInt8 ExposureIterations{ "ExposureIterations", 1 };
  prtyBoolean NoiseReduction{ "NoiseReduction", false };
  prtyFloat ApertureSize{ "ApertureSize", 0.0f };
  prtyFloat FieldOfView{ "FieldOfView", 30.0f };
  prtyFloat FocalDistance{ "FocalDistance", 0.0f };

  CCamera* m_camera;

private:
  void update();
  void updatePropsFromCamera();
};
