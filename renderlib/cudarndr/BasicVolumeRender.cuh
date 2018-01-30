#pragma once

#include "Defines.h"

void RayMarchVolume(float* outbuf, cudaTextureObject_t volumeTex, cudaTextureObject_t gradientVolumeTex, int w, int h, float density, float brightness, float* invViewMatrix, float texmin, float texmax);
void ToneMap_Basic(float* inbuf, cudaSurfaceObject_t surfaceObj, int w, int h);
