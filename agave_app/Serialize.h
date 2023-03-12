#pragma once

#include "renderlib/json/json.hpp"

#include <array>
#include <string>
#include <vector>

struct PathTraceSettings_V1
{
  float primaryStepSize;
  float secondaryStepSize;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(PathTraceSettings_V1, primaryStepSize, secondaryStepSize)
};
struct TimelineSettings_V1
{
  float minTime;
  float maxTime;
  float currentTime;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(TimelineSettings_V1, minTime, maxTime, currentTime)
};
struct CameraSettings_V1
{
  std::array<float, 3> eye;
  std::array<float, 3> target;
  std::array<float, 3> up;
  int projection;
  float fovY;
  float orthoScale;
  float exposure;
  float aperture;
  float focalDistance;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(CameraSettings_V1,
                                 eye,
                                 target,
                                 up,
                                 projection,
                                 fovY,
                                 orthoScale,
                                 exposure,
                                 aperture,
                                 focalDistance)
};
struct ControlPointSettings_V1
{
  float x;
  std::array<float, 4> value;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(ControlPointSettings_V1, x, value)
};
struct LutParams_V1
{
  float window;
  float level;
  float isovalue;
  float isorange;
  float pctLow;
  float pctHigh;
  std::vector<ControlPointSettings_V1> controlPoints;
  int mode;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(LutParams_V1, window, level, isovalue, isorange, pctLow, pctHigh, controlPoints, mode)
};
struct ChannelSettings_V1
{
  bool enabled;
  std::array<float, 3> diffuseColor;
  std::array<float, 3> specularColor;
  std::array<float, 3> emissiveColor;
  float glossiness;
  float opacity;
  LutParams_V1 lutParams;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(ChannelSettings_V1,
                                 enabled,
                                 diffuseColor,
                                 specularColor,
                                 emissiveColor,
                                 glossiness,
                                 opacity,
                                 lutParams)
};

struct LightSettings
{
  int type;
  float distance;
  float theta;
  float phi;
  std::array<float, 3> color;
  float colorIntensity;
  std::array<float, 3> topColor;
  float topColorIntensity;
  std::array<float, 3> middleColor;
  float middleColorIntensity;
  std::array<float, 3> bottomColor;
  float bottomColorIntensity;
  float width;
  float height;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(LightSettings,
                                 type,
                                 distance,
                                 theta,
                                 phi,
                                 color,
                                 colorIntensity,
                                 topColor,
                                 topColorIntensity,
                                 middleColor,
                                 middleColorIntensity,
                                 bottomColor,
                                 bottomColorIntensity,
                                 width,
                                 height)
};

struct CaptureSettings_V1
{
  int width;
  int height;
  std::string filenamePrefix;
  std::string outputDirectory;
  int samples;
  float seconds;
  int durationType;
  int startTime;
  int endTime;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(CaptureSettings_V1,
                                 width,
                                 height,
                                 filenamePrefix,
                                 outputDirectory,
                                 samples,
                                 seconds,
                                 durationType,
                                 startTime,
                                 endTime)
};

struct ViewerState_V1
{
  std::string name; // m_volumeImageFile
  // the version of this schema
  // use app version
  std::array<uint32_t, 3> version;
  std::array<int, 2> resolution; // m_resolutionX, m_resolutionY
  int renderIterations;          // m_renderIterations

  PathTraceSettings_V1 pathTracer; // m_primaryStepSize, m_secondaryStepSize
  TimelineSettings_V1 timeline;    // m_minTime, m_maxTime, m_currentTime
  int scene;                       // m_currentScene

  // [[xm, xM], [ym, yM], [zm, zM]]
  std::array<std::array<float, 2>, 3> clipRegion;

  std::array<float, 3> scale; // m_scaleX, m_scaleY, m_scaleZ

  CameraSettings_V1 camera;

  std::array<float, 3> backgroundColor; // m_backgroundColor

  std::array<float, 3> boundingBoxColor; // m_boundingBoxColor

  bool showBoundingBox; // m_showBoundingBox

  std::vector<ChannelSettings_V1> channels; // m_channels

  float density;

  // lighting
  std::vector<LightSettings> lights; // m_lights

  CaptureSettings_V1 capture;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(ViewerState_V1,
                                 name,
                                 version,
                                 resolution,
                                 renderIterations,
                                 pathTracer,
                                 timeline,
                                 scene,
                                 clipRegion,
                                 scale,
                                 camera,
                                 backgroundColor,
                                 boundingBoxColor,
                                 showBoundingBox,
                                 channels,
                                 density,
                                 lights,
                                 capture)
};
