#include <catch2/catch_test_macros.hpp>

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
T*
testcodec(const TD& data)
{
  T* cmd = new T(data);

  Command* cmdout = codec(cmd);

  T* out = dynamic_cast<T*>(cmdout);
  REQUIRE(out != nullptr);
  return out;
}

TEST_CASE("Commands can write and read from binary", "[command]")
{

  SECTION("SessionCommand")
  {
    SessionCommandD data({ "test" });
    auto cmd = testcodec<SessionCommand, SessionCommandD>(data);
    REQUIRE(cmd->toPythonString() == "session(\"test\")");
    REQUIRE(cmd->m_data.m_name == data.m_name);
  }
  SECTION("AssetPathCommand")
  {
    AssetPathCommandD data = { "test" };
    auto cmd = testcodec<AssetPathCommand, AssetPathCommandD>(data);
    REQUIRE(cmd->toPythonString() == "asset_path(\"test\")");
    REQUIRE(cmd->m_data.m_name == data.m_name);
  }
  SECTION("LoadOmeTifCommand")
  {
    LoadOmeTifCommandD data = { "test" };
    auto cmd = testcodec<LoadOmeTifCommand, LoadOmeTifCommandD>(data);
    REQUIRE(cmd->toPythonString() == "load_ome_tif(\"test\")");
    REQUIRE(cmd->m_data.m_name == data.m_name);
  }
  SECTION("SetCameraPosCommand")
  {
    SetCameraPosCommandD data = { 1.0f, 2.0f, 3.0f };
    auto cmd = testcodec<SetCameraPosCommand, SetCameraPosCommandD>(data);
    REQUIRE(cmd->toPythonString() == "eye(1, 2, 3)");
    REQUIRE(cmd->m_data.m_x == data.m_x);
    REQUIRE(cmd->m_data.m_y == data.m_y);
    REQUIRE(cmd->m_data.m_z == data.m_z);
  }
  SECTION("SetCameraTargetCommand")
  {
    SetCameraTargetCommandD data = { 1.0f, 2.0f, 3.0f };
    auto cmd = testcodec<SetCameraTargetCommand, SetCameraTargetCommandD>(data);
    REQUIRE(cmd->toPythonString() == "target(1, 2, 3)");
    REQUIRE(cmd->m_data.m_x == data.m_x);
    REQUIRE(cmd->m_data.m_y == data.m_y);
    REQUIRE(cmd->m_data.m_z == data.m_z);
  }
  SECTION("SetCameraUpCommand")
  {
    SetCameraUpCommandD data = { 1.0f, 2.0f, 3.0f };
    auto cmd = testcodec<SetCameraUpCommand, SetCameraUpCommandD>(data);
    REQUIRE(cmd->toPythonString() == "up(1, 2, 3)");
    REQUIRE(cmd->m_data.m_x == data.m_x);
    REQUIRE(cmd->m_data.m_y == data.m_y);
    REQUIRE(cmd->m_data.m_z == data.m_z);
  }
  SECTION("SetCameraApertureCommand")
  {
    SetCameraApertureCommandD data = { 1.0f };
    auto cmd = testcodec<SetCameraApertureCommand, SetCameraApertureCommandD>(data);
    REQUIRE(cmd->toPythonString() == "aperture(1)");
    REQUIRE(cmd->m_data.m_x == data.m_x);
  }
  SECTION("SetCameraProjectionCommand")
  {
    SetCameraProjectionCommandD data = { 1, 1.0f };
    auto cmd = testcodec<SetCameraProjectionCommand, SetCameraProjectionCommandD>(data);
    REQUIRE(cmd->toPythonString() == "camera_projection(1, 1)");
    REQUIRE(cmd->m_data.m_projectionType == data.m_projectionType);
    REQUIRE(cmd->m_data.m_x == data.m_x);
  }
  SECTION("SetCameraFocalDistanceCommand")
  {
    SetCameraFocalDistanceCommandD data = { 1.0f };
    auto cmd = testcodec<SetCameraFocalDistanceCommand, SetCameraFocalDistanceCommandD>(data);
    REQUIRE(cmd->toPythonString() == "focaldist(1)");
    REQUIRE(cmd->m_data.m_x == data.m_x);
  }
  SECTION("SetCameraExposureCommand")
  {
    SetCameraExposureCommandD data = { 1.0f };
    auto cmd = testcodec<SetCameraExposureCommand, SetCameraExposureCommandD>(data);
    REQUIRE(cmd->toPythonString() == "exposure(1)");
    REQUIRE(cmd->m_data.m_x == data.m_x);
  }
  SECTION("SetDiffuseColorCommand")
  {
    SetDiffuseColorCommandD data = { 2, 1.0f, 0.5f, 0.25f, 0.33f };
    auto cmd = testcodec<SetDiffuseColorCommand, SetDiffuseColorCommandD>(data);
    REQUIRE(cmd->toPythonString() == "mat_diffuse(2, 1, 0.5, 0.25, 0.33)");
    REQUIRE(cmd->m_data.m_channel == data.m_channel);
    REQUIRE(cmd->m_data.m_r == data.m_r);
    REQUIRE(cmd->m_data.m_g == data.m_g);
    REQUIRE(cmd->m_data.m_b == data.m_b);
    REQUIRE(cmd->m_data.m_a == data.m_a);
  }
  SECTION("SetSpecularColorCommand")
  {
    SetSpecularColorCommandD data = { 2, 1.0f, 0.5f, 0.25f, 0.33f };
    auto cmd = testcodec<SetSpecularColorCommand, SetSpecularColorCommandD>(data);
    REQUIRE(cmd->toPythonString() == "mat_specular(2, 1, 0.5, 0.25, 0.33)");
    REQUIRE(cmd->m_data.m_channel == data.m_channel);
    REQUIRE(cmd->m_data.m_r == data.m_r);
    REQUIRE(cmd->m_data.m_g == data.m_g);
    REQUIRE(cmd->m_data.m_b == data.m_b);
    REQUIRE(cmd->m_data.m_a == data.m_a);
  }
  SECTION("SetEmissiveColorCommand")
  {
    SetEmissiveColorCommandD data = { 2, 1.0f, 0.5f, 0.25f, 0.33f };
    auto cmd = testcodec<SetEmissiveColorCommand, SetEmissiveColorCommandD>(data);
    REQUIRE(cmd->toPythonString() == "mat_emissive(2, 1, 0.5, 0.25, 0.33)");
    REQUIRE(cmd->m_data.m_channel == data.m_channel);
    REQUIRE(cmd->m_data.m_r == data.m_r);
    REQUIRE(cmd->m_data.m_g == data.m_g);
    REQUIRE(cmd->m_data.m_b == data.m_b);
    REQUIRE(cmd->m_data.m_a == data.m_a);
  }
  SECTION("SetRenderIterationsCommand")
  {
    SetRenderIterationsCommandD data = { 2 };
    auto cmd = testcodec<SetRenderIterationsCommand, SetRenderIterationsCommandD>(data);
    REQUIRE(cmd->toPythonString() == "render_iterations(2)");
    REQUIRE(cmd->m_data.m_x == data.m_x);
  }
  SECTION("SetStreamModeCommand")
  {
    SetStreamModeCommandD data = { 2 };
    auto cmd = testcodec<SetStreamModeCommand, SetStreamModeCommandD>(data);
    REQUIRE(cmd->toPythonString() == "stream_mode(2)");
    REQUIRE(cmd->m_data.m_x == data.m_x);
  }
  SECTION("RequestRedrawCommand")
  {
    RequestRedrawCommandD data = {};
    auto cmd = testcodec<RequestRedrawCommand, RequestRedrawCommandD>(data);
    REQUIRE(cmd->toPythonString() == "redraw()");
  }
  SECTION("SetResolutionCommand")
  {
    SetResolutionCommandD data = { 200, 300 };
    auto cmd = testcodec<SetResolutionCommand, SetResolutionCommandD>(data);
    REQUIRE(cmd->toPythonString() == "set_resolution(200, 300)");
    REQUIRE(cmd->m_data.m_x == data.m_x);
    REQUIRE(cmd->m_data.m_y == data.m_y);
  }
  SECTION("SetDensityCommand")
  {
    SetDensityCommandD data = { 2.0f };
    auto cmd = testcodec<SetDensityCommand, SetDensityCommandD>(data);
    REQUIRE(cmd->toPythonString() == "density(2)");
    REQUIRE(cmd->m_data.m_x == data.m_x);
  }
  SECTION("FrameSceneCommand")
  {
    FrameSceneCommandD data = {};
    auto cmd = testcodec<FrameSceneCommand, FrameSceneCommandD>(data);
    REQUIRE(cmd->toPythonString() == "frame_scene()");
  }
  SECTION("SetGlossinessCommand")
  {
    SetGlossinessCommandD data = { 2, 2.0f };
    auto cmd = testcodec<SetGlossinessCommand, SetGlossinessCommandD>(data);
    REQUIRE(cmd->toPythonString() == "mat_glossiness(2, 2)");
    REQUIRE(cmd->m_data.m_channel == data.m_channel);
    REQUIRE(cmd->m_data.m_glossiness == data.m_glossiness);
  }
  SECTION("EnableChannelCommand")
  {
    EnableChannelCommandD data = { 2, 1 };
    auto cmd = testcodec<EnableChannelCommand, EnableChannelCommandD>(data);
    REQUIRE(cmd->toPythonString() == "enable_channel(2, 1)");
    REQUIRE(cmd->m_data.m_channel == data.m_channel);
    REQUIRE(cmd->m_data.m_enabled == data.m_enabled);
  }
  SECTION("SetWindowLevelCommand")
  {
    SetWindowLevelCommandD data = { 2, 0.25, 0.5 };
    auto cmd = testcodec<SetWindowLevelCommand, SetWindowLevelCommandD>(data);
    REQUIRE(cmd->toPythonString() == "set_window_level(2, 0.25, 0.5)");
    REQUIRE(cmd->m_data.m_channel == data.m_channel);
    REQUIRE(cmd->m_data.m_window == data.m_window);
    REQUIRE(cmd->m_data.m_level == data.m_level);
  }
  SECTION("OrbitCameraCommand")
  {
    OrbitCameraCommandD data = { 0.25, 0.5 };
    auto cmd = testcodec<OrbitCameraCommand, OrbitCameraCommandD>(data);
    REQUIRE(cmd->toPythonString() == "orbit_camera(0.25, 0.5)");
    REQUIRE(cmd->m_data.m_theta == data.m_theta);
    REQUIRE(cmd->m_data.m_phi == data.m_phi);
  }
  SECTION("SetSkylightTopColorCommand")
  {
    SetSkylightTopColorCommandD data = { 0.25, 0.5, 0.333 };
    auto cmd = testcodec<SetSkylightTopColorCommand, SetSkylightTopColorCommandD>(data);
    REQUIRE(cmd->toPythonString() == "skylight_top_color(0.25, 0.5, 0.333)");
    REQUIRE(cmd->m_data.m_r == data.m_r);
    REQUIRE(cmd->m_data.m_g == data.m_g);
    REQUIRE(cmd->m_data.m_b == data.m_b);
  }
  SECTION("SetSkylightMiddleColorCommand")
  {
    SetSkylightMiddleColorCommandD data = { 0.25, 0.5, 0.333 };
    auto cmd = testcodec<SetSkylightMiddleColorCommand, SetSkylightMiddleColorCommandD>(data);
    REQUIRE(cmd->toPythonString() == "skylight_middle_color(0.25, 0.5, 0.333)");
    REQUIRE(cmd->m_data.m_r == data.m_r);
    REQUIRE(cmd->m_data.m_g == data.m_g);
    REQUIRE(cmd->m_data.m_b == data.m_b);
  }
  SECTION("SetSkylightBottomColorCommand")
  {
    SetSkylightBottomColorCommandD data = { 0.25, 0.5, 0.333 };
    auto cmd = testcodec<SetSkylightBottomColorCommand, SetSkylightBottomColorCommandD>(data);
    REQUIRE(cmd->toPythonString() == "skylight_bottom_color(0.25, 0.5, 0.333)");
    REQUIRE(cmd->m_data.m_r == data.m_r);
    REQUIRE(cmd->m_data.m_g == data.m_g);
    REQUIRE(cmd->m_data.m_b == data.m_b);
  }
  SECTION("SetLightPosCommand")
  {
    SetLightPosCommandD data = { 1, 0.25, 0.5, 0.333 };
    auto cmd = testcodec<SetLightPosCommand, SetLightPosCommandD>(data);
    REQUIRE(cmd->toPythonString() == "light_pos(1, 0.25, 0.5, 0.333)");
    REQUIRE(cmd->m_data.m_index == data.m_index);
    REQUIRE(cmd->m_data.m_r == data.m_r);
    REQUIRE(cmd->m_data.m_theta == data.m_theta);
    REQUIRE(cmd->m_data.m_phi == data.m_phi);
  }
  SECTION("SetLightColorCommand")
  {
    SetLightColorCommandD data = { 1, 0.25, 0.5, 0.333 };
    auto cmd = testcodec<SetLightColorCommand, SetLightColorCommandD>(data);
    REQUIRE(cmd->toPythonString() == "light_color(1, 0.25, 0.5, 0.333)");
    REQUIRE(cmd->m_data.m_index == data.m_index);
    REQUIRE(cmd->m_data.m_r == data.m_r);
    REQUIRE(cmd->m_data.m_g == data.m_g);
    REQUIRE(cmd->m_data.m_b == data.m_b);
  }
  SECTION("SetLightSizeCommand")
  {
    SetLightSizeCommandD data = { 1, 0.25, 0.5 };
    auto cmd = testcodec<SetLightSizeCommand, SetLightSizeCommandD>(data);
    REQUIRE(cmd->toPythonString() == "light_size(1, 0.25, 0.5)");
    REQUIRE(cmd->m_data.m_index == data.m_index);
    REQUIRE(cmd->m_data.m_x == data.m_x);
    REQUIRE(cmd->m_data.m_y == data.m_y);
  }
  SECTION("SetClipRegionCommand")
  {
    SetClipRegionCommandD data = { 0.1, 0.9, 0.25, 0.75, 0.33333, 0.66666 };
    auto cmd = testcodec<SetClipRegionCommand, SetClipRegionCommandD>(data);
    REQUIRE(cmd->toPythonString() == "set_clip_region(0.1, 0.9, 0.25, 0.75, 0.33333, 0.66666)");
    REQUIRE(cmd->m_data.m_minx == data.m_minx);
    REQUIRE(cmd->m_data.m_maxx == data.m_maxx);
    REQUIRE(cmd->m_data.m_miny == data.m_miny);
    REQUIRE(cmd->m_data.m_maxy == data.m_maxy);
    REQUIRE(cmd->m_data.m_minz == data.m_minz);
    REQUIRE(cmd->m_data.m_maxz == data.m_maxz);
  }
  SECTION("SetVoxelScaleCommand")
  {
    SetVoxelScaleCommandD data = { 2.1, 11.9, 0.25 };
    auto cmd = testcodec<SetVoxelScaleCommand, SetVoxelScaleCommandD>(data);
    REQUIRE(cmd->toPythonString() == "set_voxel_scale(2.1, 11.9, 0.25)");
    REQUIRE(cmd->m_data.m_x == data.m_x);
    REQUIRE(cmd->m_data.m_y == data.m_y);
    REQUIRE(cmd->m_data.m_z == data.m_z);
  }
  SECTION("AutoThresholdCommand")
  {
    AutoThresholdCommandD data = { 3, 4 };
    auto cmd = testcodec<AutoThresholdCommand, AutoThresholdCommandD>(data);
    REQUIRE(cmd->toPythonString() == "auto_threshold(3, 4)");
    REQUIRE(cmd->m_data.m_channel == data.m_channel);
    REQUIRE(cmd->m_data.m_method == data.m_method);
  }
  SECTION("SetPercentileThresholdCommand")
  {
    SetPercentileThresholdCommandD data = { 3, 0.59, 0.78 };
    auto cmd = testcodec<SetPercentileThresholdCommand, SetPercentileThresholdCommandD>(data);
    REQUIRE(cmd->toPythonString() == "set_percentile_threshold(3, 0.59, 0.78)");
    REQUIRE(cmd->m_data.m_channel == data.m_channel);
    REQUIRE(cmd->m_data.m_pctLow == data.m_pctLow);
    REQUIRE(cmd->m_data.m_pctHigh == data.m_pctHigh);
  }
  SECTION("SetOpacityCommand")
  {
    SetOpacityCommandD data = { 3, 0.598 };
    auto cmd = testcodec<SetOpacityCommand, SetOpacityCommandD>(data);
    REQUIRE(cmd->toPythonString() == "mat_opacity(3, 0.598)");
    REQUIRE(cmd->m_data.m_channel == data.m_channel);
    REQUIRE(cmd->m_data.m_opacity == data.m_opacity);
  }
  SECTION("SetPrimaryRayStepSizeCommand")
  {
    SetPrimaryRayStepSizeCommandD data = { 1.25 };
    auto cmd = testcodec<SetPrimaryRayStepSizeCommand, SetPrimaryRayStepSizeCommandD>(data);
    REQUIRE(cmd->toPythonString() == "set_primary_ray_step_size(1.25)");
    REQUIRE(cmd->m_data.m_stepSize == data.m_stepSize);
  }
  SECTION("SetSecondaryRayStepSizeCommand")
  {
    SetSecondaryRayStepSizeCommandD data = { 2.25 };
    auto cmd = testcodec<SetSecondaryRayStepSizeCommand, SetSecondaryRayStepSizeCommandD>(data);
    REQUIRE(cmd->toPythonString() == "set_secondary_ray_step_size(2.25)");
    REQUIRE(cmd->m_data.m_stepSize == data.m_stepSize);
  }
  SECTION("SetBackgroundColorCommand")
  {
    SetBackgroundColorCommandD data = { 0.25, 0.5, 0.333 };
    auto cmd = testcodec<SetBackgroundColorCommand, SetBackgroundColorCommandD>(data);
    REQUIRE(cmd->toPythonString() == "background_color(0.25, 0.5, 0.333)");
    REQUIRE(cmd->m_data.m_r == data.m_r);
    REQUIRE(cmd->m_data.m_g == data.m_g);
    REQUIRE(cmd->m_data.m_b == data.m_b);
  }
  SECTION("SetIsovalueThresholdCommand")
  {
    SetIsovalueThresholdCommandD data = { 3, 0.25, 0.5 };
    auto cmd = testcodec<SetIsovalueThresholdCommand, SetIsovalueThresholdCommandD>(data);
    REQUIRE(cmd->toPythonString() == "set_isovalue_threshold(3, 0.25, 0.5)");
    REQUIRE(cmd->m_data.m_channel == data.m_channel);
    REQUIRE(cmd->m_data.m_isorange == data.m_isorange);
    REQUIRE(cmd->m_data.m_isovalue == data.m_isovalue);
  }
  SECTION("SetControlPointsCommand")
  {
    SetControlPointsCommandD data = { 3, { 0.25, 0.5, 0.333, 0.666, 0.999 } };
    auto cmd = testcodec<SetControlPointsCommand, SetControlPointsCommandD>(data);
    REQUIRE(cmd->toPythonString() == "set_control_points(3, [0.25, 0.5, 0.333, 0.666, 0.999])");
    REQUIRE(cmd->m_data.m_channel == data.m_channel);
    REQUIRE(cmd->m_data.m_data == data.m_data);
  }
  SECTION("LoadVolumeFromFileCommand")
  {
    LoadVolumeFromFileCommandD data = { "testfile", 3, 125 };
    auto cmd = testcodec<LoadVolumeFromFileCommand, LoadVolumeFromFileCommandD>(data);
    REQUIRE(cmd->toPythonString() == "load_volume_from_file(\"testfile\", 3, 125)");
    REQUIRE(cmd->m_data.m_path == data.m_path);
    REQUIRE(cmd->m_data.m_scene == data.m_scene);
    REQUIRE(cmd->m_data.m_time == data.m_time);
  }
  SECTION("SetTimeCommand")
  {
    SetTimeCommandD data = { 125 };
    auto cmd = testcodec<SetTimeCommand, SetTimeCommandD>(data);
    REQUIRE(cmd->toPythonString() == "set_time(125)");
    REQUIRE(cmd->m_data.m_time == data.m_time);
  }
  SECTION("SetBoundingBoxColorCommand")
  {
    SetBoundingBoxColorCommandD data = { 0.25, 0.5, 0.333 };
    auto cmd = testcodec<SetBoundingBoxColorCommand, SetBoundingBoxColorCommandD>(data);
    REQUIRE(cmd->toPythonString() == "bounding_box_color(0.25, 0.5, 0.333)");
    REQUIRE(cmd->m_data.m_r == data.m_r);
    REQUIRE(cmd->m_data.m_g == data.m_g);
    REQUIRE(cmd->m_data.m_b == data.m_b);
  }
  SECTION("ShowBoundingBoxCommand")
  {
    ShowBoundingBoxCommandD data = { 1 };
    auto cmd = testcodec<ShowBoundingBoxCommand, ShowBoundingBoxCommandD>(data);
    REQUIRE(cmd->toPythonString() == "show_bounding_box(1)");
    REQUIRE(cmd->m_data.m_on == data.m_on);
  }
  SECTION("TrackballCameraCommand")
  {
    TrackballCameraCommandD data = { 0.25, 0.5 };
    auto cmd = testcodec<TrackballCameraCommand, TrackballCameraCommandD>(data);
    REQUIRE(cmd->toPythonString() == "trackball_camera(0.25, 0.5)");
    REQUIRE(cmd->m_data.m_theta == data.m_theta);
    REQUIRE(cmd->m_data.m_phi == data.m_phi);
  }
  SECTION("LoadSetSourceCommand")
  {
    LoadSetSourceCommandD data = { "testfile", 3, 4 };
    auto cmd = testcodec<LoadSetSourceCommand, LoadSetSourceCommandD>(data);
    REQUIRE(cmd->toPythonString() == "load_set_source(\"testfile\", 3, 4)");
    REQUIRE(cmd->m_data.m_path == data.m_path);
    REQUIRE(cmd->m_data.m_scene == data.m_scene);
    REQUIRE(cmd->m_data.m_level == data.m_level);
  }
  SECTION("LoadSetChannelsCommand")
  {
    LoadSetChannelsCommandD data = { { 0, 1, 2, 3, 4 } };
    auto cmd = testcodec<LoadSetChannelsCommand, LoadSetChannelsCommandD>(data);
    REQUIRE(cmd->toPythonString() == "load_set_channels([0, 1, 2, 3, 4])");
    REQUIRE(cmd->m_data.m_channels == data.m_channels);
  }
  SECTION("LoadSetRegionCommand")
  {
    LoadSetRegionCommandD data = { 0, 1, 2, 3, 4, 5 };
    auto cmd = testcodec<LoadSetRegionCommand, LoadSetRegionCommandD>(data);
    REQUIRE(cmd->toPythonString() == "load_set_region(0, 1, 2, 3, 4, 5)");
    REQUIRE(cmd->m_data.m_xmin == data.m_xmin);
    REQUIRE(cmd->m_data.m_xmax == data.m_xmax);
    REQUIRE(cmd->m_data.m_ymin == data.m_ymin);
    REQUIRE(cmd->m_data.m_ymax == data.m_ymax);
    REQUIRE(cmd->m_data.m_zmin == data.m_zmin);
    REQUIRE(cmd->m_data.m_zmax == data.m_zmax);
  }
}
