#include "catch.hpp"

#include "../agave_app/commandBuffer.h"
#include "renderlib/command.h"

#include <vector>

Command*
codec(Command* cmd)
{
  std::vector<Command*> cmds({ cmd });
  commandBuffer* buffer = commandBuffer::createBuffer(cmds);
  buffer->processBuffer();
  std::vector<Command*> out = buffer->getQueue();
  REQUIRE(out.size() == 1);
  delete buffer;
  return out[0];
}

template<typename T, typename TD>
TD
testcodec(const TD& data)
{
  T* cmd = new T(data);

  Command* cmdout = codec(cmd);

  T* out = dynamic_cast<T*>(cmdout);
  REQUIRE(out != nullptr);
  return out->m_data;
}

TEST_CASE("Commands can write and read from binary", "[command]")
{

  SECTION("SessionCommand")
  {
    SessionCommandD data({ "test" });
    auto outdata = testcodec<SessionCommand, SessionCommandD>(data);
    REQUIRE(outdata.m_name == data.m_name);
  }
  SECTION("AssetPathCommand")
  {
    AssetPathCommandD data = { "test" };
    auto outdata = testcodec<AssetPathCommand, AssetPathCommandD>(data);
    REQUIRE(outdata.m_name == data.m_name);
  }
  SECTION("LoadOmeTifCommand")
  {
    LoadOmeTifCommandD data = { "test" };
    auto outdata = testcodec<LoadOmeTifCommand, LoadOmeTifCommandD>(data);
    REQUIRE(outdata.m_name == data.m_name);
  }
  SECTION("SetCameraPosCommand")
  {
    SetCameraPosCommandD data = { 1.0f, 2.0f, 3.0f };
    auto outdata = testcodec<SetCameraPosCommand, SetCameraPosCommandD>(data);
    REQUIRE(outdata.m_x == data.m_x);
    REQUIRE(outdata.m_y == data.m_y);
    REQUIRE(outdata.m_z == data.m_z);
  }
  SECTION("SetCameraTargetCommand")
  {
    SetCameraTargetCommandD data = { 1.0f, 2.0f, 3.0f };
    auto outdata = testcodec<SetCameraTargetCommand, SetCameraTargetCommandD>(data);
    REQUIRE(outdata.m_x == data.m_x);
    REQUIRE(outdata.m_y == data.m_y);
    REQUIRE(outdata.m_z == data.m_z);
  }
  SECTION("SetCameraUpCommand")
  {
    SetCameraUpCommandD data = { 1.0f, 2.0f, 3.0f };
    auto outdata = testcodec<SetCameraUpCommand, SetCameraUpCommandD>(data);
    REQUIRE(outdata.m_x == data.m_x);
    REQUIRE(outdata.m_y == data.m_y);
    REQUIRE(outdata.m_z == data.m_z);
  }
  SECTION("SetCameraApertureCommand")
  {
    SetCameraApertureCommandD data = { 1.0f };
    auto outdata = testcodec<SetCameraApertureCommand, SetCameraApertureCommandD>(data);
    REQUIRE(outdata.m_x == data.m_x);
  }
  SECTION("SetCameraProjectionCommand")
  {
    SetCameraProjectionCommandD data = { 1, 1.0f };
    auto outdata = testcodec<SetCameraProjectionCommand, SetCameraProjectionCommandD>(data);
    REQUIRE(outdata.m_projectionType == data.m_projectionType);
    REQUIRE(outdata.m_x == data.m_x);
  }
  SECTION("SetCameraFocalDistanceCommand")
  {
    SetCameraFocalDistanceCommandD data = { 1.0f };
    auto outdata = testcodec<SetCameraFocalDistanceCommand, SetCameraFocalDistanceCommandD>(data);
    REQUIRE(outdata.m_x == data.m_x);
  }
  SECTION("SetCameraExposureCommand")
  {
    SetCameraExposureCommandD data = { 1.0f };
    auto outdata = testcodec<SetCameraExposureCommand, SetCameraExposureCommandD>(data);
    REQUIRE(outdata.m_x == data.m_x);
  }
  SECTION("SetDiffuseColorCommand")
  {
    SetDiffuseColorCommandD data = { 2, 1.0f, 0.5f, 0.25f, 0.33f };
    auto outdata = testcodec<SetDiffuseColorCommand, SetDiffuseColorCommandD>(data);
    REQUIRE(outdata.m_channel == data.m_channel);
    REQUIRE(outdata.m_r == data.m_r);
    REQUIRE(outdata.m_g == data.m_g);
    REQUIRE(outdata.m_b == data.m_b);
    REQUIRE(outdata.m_a == data.m_a);
  }
  SECTION("SetSpecularColorCommand")
  {
    SetSpecularColorCommandD data = { 2, 1.0f, 0.5f, 0.25f, 0.33f };
    auto outdata = testcodec<SetSpecularColorCommand, SetSpecularColorCommandD>(data);
    REQUIRE(outdata.m_channel == data.m_channel);
    REQUIRE(outdata.m_r == data.m_r);
    REQUIRE(outdata.m_g == data.m_g);
    REQUIRE(outdata.m_b == data.m_b);
    REQUIRE(outdata.m_a == data.m_a);
  }
  SECTION("SetEmissiveColorCommand")
  {
    SetEmissiveColorCommandD data = { 2, 1.0f, 0.5f, 0.25f, 0.33f };
    auto outdata = testcodec<SetEmissiveColorCommand, SetEmissiveColorCommandD>(data);
    REQUIRE(outdata.m_channel == data.m_channel);
    REQUIRE(outdata.m_r == data.m_r);
    REQUIRE(outdata.m_g == data.m_g);
    REQUIRE(outdata.m_b == data.m_b);
    REQUIRE(outdata.m_a == data.m_a);
  }
  SECTION("SetRenderIterationsCommand")
  {
    SetRenderIterationsCommandD data = { 2 };
    auto outdata = testcodec<SetRenderIterationsCommand, SetRenderIterationsCommandD>(data);
    REQUIRE(outdata.m_x == data.m_x);
  }
  SECTION("SetStreamModeCommand")
  {
    SetStreamModeCommandD data = { 2 };
    auto outdata = testcodec<SetStreamModeCommand, SetStreamModeCommandD>(data);
    REQUIRE(outdata.m_x == data.m_x);
  }
  SECTION("RequestRedrawCommand")
  {
    RequestRedrawCommandD data = {};
    auto outdata = testcodec<RequestRedrawCommand, RequestRedrawCommandD>(data);
  }
  SECTION("SetResolutionCommand")
  {
    SetResolutionCommandD data = { 200, 300 };
    auto outdata = testcodec<SetResolutionCommand, SetResolutionCommandD>(data);
    REQUIRE(outdata.m_x == data.m_x);
    REQUIRE(outdata.m_y == data.m_y);
  }
  SECTION("SetDensityCommand")
  {
    SetDensityCommandD data = { 2.0f };
    auto outdata = testcodec<SetDensityCommand, SetDensityCommandD>(data);
    REQUIRE(outdata.m_x == data.m_x);
  }
  SECTION("FrameSceneCommand")
  {
    FrameSceneCommandD data = {};
    auto outdata = testcodec<FrameSceneCommand, FrameSceneCommandD>(data);
  }
  SECTION("SetGlossinessCommand")
  {
    SetGlossinessCommandD data = { 2, 2.0f };
    auto outdata = testcodec<SetGlossinessCommand, SetGlossinessCommandD>(data);
    REQUIRE(outdata.m_channel == data.m_channel);
    REQUIRE(outdata.m_glossiness == data.m_glossiness);
  }
  SECTION("EnableChannelCommand")
  {
    EnableChannelCommandD data = { 2, 1 };
    auto outdata = testcodec<EnableChannelCommand, EnableChannelCommandD>(data);
    REQUIRE(outdata.m_channel == data.m_channel);
    REQUIRE(outdata.m_enabled == data.m_enabled);
  }
  SECTION("SetWindowLevelCommand")
  {
    SetWindowLevelCommandD data = { 2, 0.25, 0.5 };
    auto outdata = testcodec<SetWindowLevelCommand, SetWindowLevelCommandD>(data);
    REQUIRE(outdata.m_channel == data.m_channel);
    REQUIRE(outdata.m_window == data.m_window);
    REQUIRE(outdata.m_level == data.m_level);
  }
  SECTION("OrbitCameraCommand")
  {
    OrbitCameraCommandD data = { 0.25, 0.5 };
    auto outdata = testcodec<OrbitCameraCommand, OrbitCameraCommandD>(data);
    REQUIRE(outdata.m_theta == data.m_theta);
    REQUIRE(outdata.m_phi == data.m_phi);
  }
  SECTION("SetSkylightTopColorCommand")
  {
    SetSkylightTopColorCommandD data = { 0.25, 0.5, 0.333 };
    auto outdata = testcodec<SetSkylightTopColorCommand, SetSkylightTopColorCommandD>(data);
    REQUIRE(outdata.m_r == data.m_r);
    REQUIRE(outdata.m_g == data.m_g);
    REQUIRE(outdata.m_b == data.m_b);
  }
  SECTION("SetSkylightMiddleColorCommand")
  {
    SetSkylightMiddleColorCommandD data = { 0.25, 0.5, 0.333 };
    auto outdata = testcodec<SetSkylightMiddleColorCommand, SetSkylightMiddleColorCommandD>(data);
    REQUIRE(outdata.m_r == data.m_r);
    REQUIRE(outdata.m_g == data.m_g);
    REQUIRE(outdata.m_b == data.m_b);
  }
  SECTION("SetSkylightBottomColorCommand")
  {
    SetSkylightBottomColorCommandD data = { 0.25, 0.5, 0.333 };
    auto outdata = testcodec<SetSkylightBottomColorCommand, SetSkylightBottomColorCommandD>(data);
    REQUIRE(outdata.m_r == data.m_r);
    REQUIRE(outdata.m_g == data.m_g);
    REQUIRE(outdata.m_b == data.m_b);
  }
  SECTION("SetLightPosCommand")
  {
    SetLightPosCommandD data = { 1, 0.25, 0.5, 0.333 };
    auto outdata = testcodec<SetLightPosCommand, SetLightPosCommandD>(data);
    REQUIRE(outdata.m_index == data.m_index);
    REQUIRE(outdata.m_r == data.m_r);
    REQUIRE(outdata.m_theta == data.m_theta);
    REQUIRE(outdata.m_phi == data.m_phi);
  }
  SECTION("SetLightColorCommand")
  {
    SetLightColorCommandD data = { 1, 0.25, 0.5, 0.333 };
    auto outdata = testcodec<SetLightColorCommand, SetLightColorCommandD>(data);
    REQUIRE(outdata.m_index == data.m_index);
    REQUIRE(outdata.m_r == data.m_r);
    REQUIRE(outdata.m_g == data.m_g);
    REQUIRE(outdata.m_b == data.m_b);
  }
  SECTION("SetLightSizeCommand")
  {
    SetLightSizeCommandD data = { 1, 0.25, 0.5 };
    auto outdata = testcodec<SetLightSizeCommand, SetLightSizeCommandD>(data);
    REQUIRE(outdata.m_index == data.m_index);
    REQUIRE(outdata.m_x == data.m_x);
    REQUIRE(outdata.m_y == data.m_y);
  }
  SECTION("SetClipRegionCommand")
  {
    SetClipRegionCommandD data = { 0.1, 0.9, 0.25, 0.75, 0.33333, 0.66666 };
    auto outdata = testcodec<SetClipRegionCommand, SetClipRegionCommandD>(data);
    REQUIRE(outdata.m_minx == data.m_minx);
    REQUIRE(outdata.m_maxx == data.m_maxx);
    REQUIRE(outdata.m_miny == data.m_miny);
    REQUIRE(outdata.m_maxy == data.m_maxy);
    REQUIRE(outdata.m_minz == data.m_minz);
    REQUIRE(outdata.m_maxz == data.m_maxz);
  }
  SECTION("SetVoxelScaleCommand")
  {
    SetVoxelScaleCommandD data = { 2.1, 11.9, 0.25 };
    auto outdata = testcodec<SetVoxelScaleCommand, SetVoxelScaleCommandD>(data);
    REQUIRE(outdata.m_x == data.m_x);
    REQUIRE(outdata.m_y == data.m_y);
    REQUIRE(outdata.m_z == data.m_z);
  }
  SECTION("AutoThresholdCommand")
  {
    AutoThresholdCommandD data = { 3, 4 };
    auto outdata = testcodec<AutoThresholdCommand, AutoThresholdCommandD>(data);
    REQUIRE(outdata.m_channel == data.m_channel);
    REQUIRE(outdata.m_method == data.m_method);
  }
  SECTION("SetPercentileThresholdCommand")
  {
    SetPercentileThresholdCommandD data = { 3, 0.59, 0.78 };
    auto outdata = testcodec<SetPercentileThresholdCommand, SetPercentileThresholdCommandD>(data);
    REQUIRE(outdata.m_channel == data.m_channel);
    REQUIRE(outdata.m_pctLow == data.m_pctLow);
    REQUIRE(outdata.m_pctHigh == data.m_pctHigh);
  }
  SECTION("SetOpacityCommand")
  {
    SetOpacityCommandD data = { 3, 0.598 };
    auto outdata = testcodec<SetOpacityCommand, SetOpacityCommandD>(data);
    REQUIRE(outdata.m_channel == data.m_channel);
    REQUIRE(outdata.m_opacity == data.m_opacity);
  }
  SECTION("SetPrimaryRayStepSizeCommand")
  {
    SetPrimaryRayStepSizeCommandD data = { 1.25 };
    auto outdata = testcodec<SetPrimaryRayStepSizeCommand, SetPrimaryRayStepSizeCommandD>(data);
    REQUIRE(outdata.m_stepSize == data.m_stepSize);
  }
  SECTION("SetSecondaryRayStepSizeCommand")
  {
    SetSecondaryRayStepSizeCommandD data = { 2.25 };
    auto outdata = testcodec<SetSecondaryRayStepSizeCommand, SetSecondaryRayStepSizeCommandD>(data);
    REQUIRE(outdata.m_stepSize == data.m_stepSize);
  }
  SECTION("SetBackgroundColorCommand")
  {
    SetBackgroundColorCommandD data = { 0.25, 0.5, 0.333 };
    auto outdata = testcodec<SetBackgroundColorCommand, SetBackgroundColorCommandD>(data);
    REQUIRE(outdata.m_r == data.m_r);
    REQUIRE(outdata.m_g == data.m_g);
    REQUIRE(outdata.m_b == data.m_b);
  }
  SECTION("SetIsovalueThresholdCommand")
  {
    SetIsovalueThresholdCommandD data = { 3, 0.25, 0.5 };
    auto outdata = testcodec<SetIsovalueThresholdCommand, SetIsovalueThresholdCommandD>(data);
    REQUIRE(outdata.m_channel == data.m_channel);
    REQUIRE(outdata.m_isorange == data.m_isorange);
    REQUIRE(outdata.m_isovalue == data.m_isovalue);
  }
  SECTION("SetControlPointsCommand")
  {
    SetControlPointsCommandD data = { 3, { 0.25, 0.5, 0.333, 0.666, 0.999 } };
    auto outdata = testcodec<SetControlPointsCommand, SetControlPointsCommandD>(data);
    REQUIRE(outdata.m_channel == data.m_channel);
    REQUIRE(outdata.m_data == data.m_data);
  }
  SECTION("LoadVolumeFromFileCommand")
  {
    LoadVolumeFromFileCommandD data = { "testfile", 3, 125 };
    auto outdata = testcodec<LoadVolumeFromFileCommand, LoadVolumeFromFileCommandD>(data);
    REQUIRE(outdata.m_path == data.m_path);
    REQUIRE(outdata.m_scene == data.m_scene);
    REQUIRE(outdata.m_time == data.m_time);
  }
  SECTION("SetTimeCommand")
  {
    SetTimeCommandD data = { 125 };
    auto outdata = testcodec<SetTimeCommand, SetTimeCommandD>(data);
    REQUIRE(outdata.m_time == data.m_time);
  }
  SECTION("SetBoundingBoxColorCommand")
  {
    SetBoundingBoxColorCommandD data = { 0.25, 0.5, 0.333 };
    auto outdata = testcodec<SetBoundingBoxColorCommand, SetBoundingBoxColorCommandD>(data);
    REQUIRE(outdata.m_r == data.m_r);
    REQUIRE(outdata.m_g == data.m_g);
    REQUIRE(outdata.m_b == data.m_b);
  }
  SECTION("ShowBoundingBoxCommand")
  {
    ShowBoundingBoxCommandD data = { 1 };
    auto outdata = testcodec<ShowBoundingBoxCommand, ShowBoundingBoxCommandD>(data);
    REQUIRE(outdata.m_on == data.m_on);
  }
  SECTION("TrackballCameraCommand")
  {
    TrackballCameraCommandD data = { 0.25, 0.5 };
    auto outdata = testcodec<TrackballCameraCommand, TrackballCameraCommandD>(data);
    REQUIRE(outdata.m_theta == data.m_theta);
    REQUIRE(outdata.m_phi == data.m_phi);
  }
}
