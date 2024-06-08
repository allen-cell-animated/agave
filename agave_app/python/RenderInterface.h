#pragma once

#include <inttypes.h>
#include <string>
#include <vector>

class RenderInterface
{
public:
  // tell server to identify this session?
  virtual int Session(const std::string&) = 0;
  // tell server where files might be (appends to existing)
  virtual int AssetPath(const std::string&) = 0;
  // load a volume
  virtual int LoadOmeTif(const std::string&) = 0;
  virtual int LoadVolumeFromFile(const std::string&, int, int) = 0;
  // change load same volume file, different time index
  virtual int SetTime(int) = 0;
  // set camera pos
  virtual int Eye(float, float, float) = 0;
  // set camera target pt
  virtual int Target(float, float, float) = 0;
  // set camera up direction
  virtual int Up(float, float, float) = 0;
  virtual int Aperture(float) = 0;
  // perspective(0)/ortho(1), fov(degrees)/orthoscale(world units)
  virtual int CameraProjection(int32_t, float) = 0;
  virtual int Focaldist(float) = 0;
  virtual int Exposure(float) = 0;
  virtual int MatDiffuse(int32_t, float, float, float, float) = 0;
  virtual int MatSpecular(int32_t, float, float, float, float) = 0;
  virtual int MatEmissive(int32_t, float, float, float, float) = 0;
  // set num render iterations
  virtual int RenderIterations(int32_t) = 0;
  // (continuous or on-demand frames)
  virtual int StreamMode(int32_t) = 0;
  // request new image
  virtual int Redraw() = 0;
  virtual int SetResolution(int32_t, int32_t) = 0;
  virtual int Density(float) = 0;
  // move camera to bound and look at the scene contents
  virtual int FrameScene() = 0;
  virtual int MatGlossiness(int32_t, float) = 0;
  // channel index, 1/0 for enable/disable
  virtual int EnableChannel(int32_t, int32_t) = 0;
  // channel index, window, level.  (Do I ever set these independently?)
  virtual int SetWindowLevel(int32_t, float, float) = 0;
  // theta, phi in degrees
  virtual int OrbitCamera(float, float) = 0;
  virtual int TrackballCamera(float, float) = 0;
  virtual int SkylightTopColor(float, float, float) = 0;
  virtual int SkylightMiddleColor(float, float, float) = 0;
  virtual int SkylightBottomColor(float, float, float) = 0;
  // r, theta, phi
  virtual int LightPos(int32_t, float, float, float) = 0;
  virtual int LightColor(int32_t, float, float, float) = 0;
  // x by y size
  virtual int LightSize(int32_t, float, float) = 0;
  // xmin, xmax, ymin, ymax, zmin, zmax
  virtual int SetClipRegion(float, float, float, float, float, float) = 0;
  // x, y, z pixel scaling
  virtual int SetVoxelScale(float, float, float) = 0;
  // channel, method
  virtual int AutoThreshold(int32_t, int32_t) = 0;
  // channel index, pct_low, pct_high.  (Do I ever set these independently?)
  virtual int SetPercentileThreshold(int32_t, float, float) = 0;
  virtual int MatOpacity(int32_t, float) = 0;
  virtual int SetPrimaryRayStepSize(float) = 0;
  virtual int SetSecondaryRayStepSize(float) = 0;
  virtual int BackgroundColor(float, float, float) = 0;
  virtual int SetIsovalueThreshold(int32_t, float, float) = 0;
  virtual int SetControlPoints(int32_t, std::vector<float>) = 0;
  virtual int SetBoundingBoxColor(float, float, float) = 0;
  virtual int ShowBoundingBox(int32_t) = 0;
  virtual int ShowScaleBar(int32_t) = 0;
  virtual int SetFlipAxis(int32_t, int32_t, int32_t) = 0;
  virtual int SetInterpolation(int32_t) = 0;
};
