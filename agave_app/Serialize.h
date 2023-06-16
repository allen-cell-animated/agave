#pragma once

#include "SerializeV1.h"
#include "renderlib/json/json.hpp"

#include <array>
#include <string>
#include <vector>

namespace Serialize {

enum class RendererType_PID : int
{
  PATHTRACE = 0,
  RAYMARCH = 1
};

struct LoadSettings
{
  std::string url;
  std::string subpath;
  uint32_t scene = 0;
  uint32_t time = 0;
  std::vector<uint32_t> channels;
  std::array<std::array<uint32_t, 2>, 3> clipRegion = { 0, 0, 0, 0, 0, 0 };

  bool operator==(const LoadSettings& other) const
  {
    return url == other.url && subpath == other.subpath && scene == other.scene && time == other.time &&
           channels == other.channels && clipRegion == other.clipRegion;
  }
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(LoadSettings, url, subpath, scene, time, channels, clipRegion)
};

enum class DurationType_PID : int
{
  SAMPLES = 0,
  TIME = 1
};

struct CaptureSettings
{
  int width = 960;
  int height = 540;
  std::string filenamePrefix = "frame";
  std::string outputDirectory = ".";
  int samples = 32;
  float seconds = 10.0f;
  DurationType_PID durationType = DurationType_PID::SAMPLES;
  int startTime = 0;
  int endTime = 0;

  bool operator==(const CaptureSettings& other) const
  {
    return width == other.width && height == other.height && filenamePrefix == other.filenamePrefix &&
           outputDirectory == other.outputDirectory && samples == other.samples && seconds == other.seconds &&
           durationType == other.durationType && startTime == other.startTime && endTime == other.endTime;
  }
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(CaptureSettings,
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

struct ViewerState
{
  std::vector<LoadSettings> datasets;

  // the app version that wrote this data
  std::array<uint32_t, 3> version{ 0, 0, 0 };

  RendererType_PID rendererType = RendererType_PID::PATHTRACE;
  PathTraceSettings_V1 pathTracer; // m_primaryStepSize, m_secondaryStepSize
  TimelineSettings_V1 timeline;    // m_minTime, m_maxTime, m_currentTime

  // [[xm, xM], [ym, yM], [zm, zM]]
  std::array<std::array<float, 2>, 3> clipRegion = { 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f };

  std::array<float, 3> scale = { 1, 1, 1 }; // m_scaleX, m_scaleY, m_scaleZ

  CameraSettings_V1 camera;

  std::array<float, 3> backgroundColor = { 0, 0, 0 }; // m_backgroundColor

  std::array<float, 3> boundingBoxColor = { 1, 1, 1 }; // m_boundingBoxColor

  bool showBoundingBox = false; // m_showBoundingBox

  std::vector<ChannelSettings_V1> channels; // m_channels

  float density = 50.0f;

  // lighting
  std::vector<LightSettings_V1> lights; // m_lights

  CaptureSettings capture;

  bool operator==(const ViewerState& other) const
  {
    return datasets == other.datasets && version == other.version && pathTracer == other.pathTracer &&
           timeline == other.timeline && clipRegion == other.clipRegion && scale == other.scale &&
           camera == other.camera && backgroundColor == other.backgroundColor &&
           boundingBoxColor == other.boundingBoxColor && showBoundingBox == other.showBoundingBox &&
           channels == other.channels && density == other.density && lights == other.lights && capture == other.capture;
  }
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(ViewerState,
                                 datasets,
                                 version,
                                 rendererType,
                                 pathTracer,
                                 timeline,
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

ViewerState
fromV1(const ViewerState_V1& v1);

} // namespace Serialize
