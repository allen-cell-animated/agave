#pragma once

#include "core/prty/prtyProperty.h"
#include "CCamera.h"

class CameraDataObject
{
public:
  CameraDataObject(CCamera* camera);

  prtyProperty<float> Exposure{ "Exposure", 0.75f };
  prtyProperty<int> ExposureIterations{ "ExposureIterations", 1 };
  prtyProperty<bool> NoiseReduction{ "NoiseReduction", false };
  prtyProperty<float> ApertureSize{ "ApertureSize", 0.0f };
  prtyProperty<float> FieldOfView{ "FieldOfView", 30.0f };
  prtyProperty<float> FocalDistance{ "FocalDistance", 0.0f };

  CCamera* m_camera;

private:
  void update();
  void updatePropsFromCamera();
};
