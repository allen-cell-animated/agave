#pragma once

#include <string>
#include <vector>

class CCamera;
class Renderer;
class RenderSettings;
class Scene;

enum CommandArgType
{
  I32,
  F32,
  STR
};

struct ExecutionContext
{
  Renderer* m_renderer;
  RenderSettings* m_renderSettings;
  Scene* m_appScene;
  CCamera* m_camera;
  std::string m_message;
};

class Command
{
public:
  // return number of bytes advanced
  virtual void execute(ExecutionContext* context) = 0;

  virtual ~Command() {}
};

#define CMDDECL(NAME, CMDID, PYTHONNAME, ARGTYPES)                                                                     \
  class NAME : public Command                                                                                          \
  {                                                                                                                    \
  public:                                                                                                              \
    NAME(NAME##D d)                                                                                                    \
      : m_data(d)                                                                                                      \
    {}                                                                                                                 \
    virtual void execute(ExecutionContext* context);                                                                   \
    static const uint32_t m_ID = CMDID;                                                                                \
    static const std::string PythonName()                                                                              \
    {                                                                                                                  \
      static const std::string s = PYTHONNAME;                                                                         \
      return s;                                                                                                        \
    }                                                                                                                  \
    static const std::vector<CommandArgType> ArgTypes()                                                                \
    {                                                                                                                  \
      static const std::vector<CommandArgType> a = ARGTYPES;                                                           \
      return a;                                                                                                        \
    }                                                                                                                  \
    NAME##D m_data;                                                                                                    \
  };

#define CMD_ARGS(...) __VA_ARGS__

struct SessionCommandD
{
  std::string m_name;
};
CMDDECL(SessionCommand, 0, "session", CMD_ARGS({ STR }));

struct AssetPathCommandD
{
  std::string m_name;
};
CMDDECL(AssetPathCommand, 1, "asset_path", CMD_ARGS({ STR }));

struct LoadOmeTifCommandD
{
  std::string m_name;
};
CMDDECL(LoadOmeTifCommand, 2, "load_ome_tif", CMD_ARGS({ STR }));

struct SetCameraPosCommandD
{
  float m_x, m_y, m_z;
};
CMDDECL(SetCameraPosCommand, 3, "eye", CMD_ARGS({ F32, F32, F32 }));

struct SetCameraTargetCommandD
{
  float m_x, m_y, m_z;
};
CMDDECL(SetCameraTargetCommand, 4, "target", CMD_ARGS({ F32, F32, F32 }));

struct SetCameraUpCommandD
{
  float m_x, m_y, m_z;
};
CMDDECL(SetCameraUpCommand, 5, "up", CMD_ARGS({ F32, F32, F32 }));

struct SetCameraApertureCommandD
{
  float m_x;
};
CMDDECL(SetCameraApertureCommand, 6, "aperture", CMD_ARGS({ F32 }));

struct SetCameraProjectionCommandD
{
  // perspective or ortho
  int32_t m_projectionType;
  // fov degrees or ortho scale
  float m_x;
};
CMDDECL(SetCameraProjectionCommand, 7, "camera_projection", CMD_ARGS({ I32, F32 }));

struct SetCameraFocalDistanceCommandD
{
  float m_x;
};
CMDDECL(SetCameraFocalDistanceCommand, 8, "focaldist", CMD_ARGS({ F32 }));

struct SetCameraExposureCommandD
{
  float m_x;
};
CMDDECL(SetCameraExposureCommand, 9, "exposure", CMD_ARGS({ F32 }));

struct SetDiffuseColorCommandD
{
  int32_t m_channel;
  float m_r, m_g, m_b, m_a;
};
CMDDECL(SetDiffuseColorCommand, 10, "mat_diffuse", CMD_ARGS({ I32, F32, F32, F32, F32 }));

struct SetSpecularColorCommandD
{
  int32_t m_channel;
  float m_r, m_g, m_b, m_a;
};
CMDDECL(SetSpecularColorCommand, 11, "mat_specular", CMD_ARGS({ I32, F32, F32, F32, F32 }));

struct SetEmissiveColorCommandD
{
  int32_t m_channel;
  float m_r, m_g, m_b, m_a;
};
CMDDECL(SetEmissiveColorCommand, 12, "mat_emissive", CMD_ARGS({ I32, F32, F32, F32, F32 }));

struct SetRenderIterationsCommandD
{
  int32_t m_x;
};
CMDDECL(SetRenderIterationsCommand, 13, "render_iterations", CMD_ARGS({ I32 }));

struct SetStreamModeCommandD
{
  int32_t m_x;
};
CMDDECL(SetStreamModeCommand, 14, "stream_mode", CMD_ARGS({ I32 }));

struct RequestRedrawCommandD
{};
CMDDECL(RequestRedrawCommand, 15, "redraw", CMD_ARGS({}));

struct SetResolutionCommandD
{
  int32_t m_x, m_y;
};
CMDDECL(SetResolutionCommand, 16, "set_resolution", CMD_ARGS({ I32, I32 }));

struct SetDensityCommandD
{
  float m_x;
};
CMDDECL(SetDensityCommand, 17, "density", CMD_ARGS({ F32 }));

struct FrameSceneCommandD
{};
CMDDECL(FrameSceneCommand, 18, "frame_scene", CMD_ARGS({}));

struct SetGlossinessCommandD
{
  int32_t m_channel;
  float m_glossiness;
};
CMDDECL(SetGlossinessCommand, 19, "mat_glossiness", CMD_ARGS({ I32, F32 }));

struct EnableChannelCommandD
{
  int32_t m_channel;
  int32_t m_enabled;
};
CMDDECL(EnableChannelCommand, 20, "enable_channel", CMD_ARGS({ I32, I32 }));

struct SetWindowLevelCommandD
{
  int32_t m_channel;
  float m_window;
  float m_level;
};
CMDDECL(SetWindowLevelCommand, 21, "set_window_level", CMD_ARGS({ I32, F32, F32 }));

struct OrbitCameraCommandD
{
  float m_theta;
  float m_phi;
};
CMDDECL(OrbitCameraCommand, 22, "orbit_camera", CMD_ARGS({ F32, F32 }));

struct SetSkylightTopColorCommandD
{
  float m_r, m_g, m_b;
};
CMDDECL(SetSkylightTopColorCommand, 23, "skylight_top_color", CMD_ARGS({ F32, F32, F32 }));

struct SetSkylightMiddleColorCommandD
{
  float m_r, m_g, m_b;
};
CMDDECL(SetSkylightMiddleColorCommand, 24, "skylight_middle_color", CMD_ARGS({ F32, F32, F32 }));

struct SetSkylightBottomColorCommandD
{
  float m_r, m_g, m_b;
};
CMDDECL(SetSkylightBottomColorCommand, 25, "skylight_bottom_color", CMD_ARGS({ F32, F32, F32 }));

struct SetLightPosCommandD
{
  int32_t m_index;
  float m_r, m_theta, m_phi;
};
CMDDECL(SetLightPosCommand, 26, "light_pos", CMD_ARGS({ I32, F32, F32, F32 }));

struct SetLightColorCommandD
{
  int32_t m_index;
  float m_r, m_g, m_b;
};
CMDDECL(SetLightColorCommand, 27, "light_color", CMD_ARGS({ I32, F32, F32, F32 }));

struct SetLightSizeCommandD
{
  int32_t m_index;
  float m_x, m_y;
};
CMDDECL(SetLightSizeCommand, 28, "light_size", CMD_ARGS({ I32, F32, F32 }));

struct SetClipRegionCommandD
{
  float m_minx, m_maxx;
  float m_miny, m_maxy;
  float m_minz, m_maxz;
};
CMDDECL(SetClipRegionCommand, 29, "set_clip_region", CMD_ARGS({ F32, F32, F32, F32, F32, F32 }));

struct SetVoxelScaleCommandD
{
  float m_x, m_y, m_z;
};
CMDDECL(SetVoxelScaleCommand, 30, "set_voxel_scale", CMD_ARGS({ F32, F32, F32 }));

struct AutoThresholdCommandD
{
  int32_t m_channel;
  int32_t m_method;
};
CMDDECL(AutoThresholdCommand, 31, "auto_threshold", CMD_ARGS({ I32, I32 }));

struct SetPercentileThresholdCommandD
{
  int32_t m_channel;
  float m_pctLow;
  float m_pctHigh;
};
CMDDECL(SetPercentileThresholdCommand, 32, "set_percentile_threshold", CMD_ARGS({ I32, F32, F32 }));

struct SetOpacityCommandD
{
  int32_t m_channel;
  float m_opacity;
};
CMDDECL(SetOpacityCommand, 33, "mat_opacity", CMD_ARGS({ I32, F32 }));

struct SetPrimaryRayStepSizeCommandD
{
  float m_stepSize;
};
CMDDECL(SetPrimaryRayStepSizeCommand, 34, "set_primary_ray_step_size", CMD_ARGS({ F32 }));

struct SetSecondaryRayStepSizeCommandD
{
  float m_stepSize;
};
CMDDECL(SetSecondaryRayStepSizeCommand, 35, "set_secondary_ray_step_size", CMD_ARGS({ F32 }));

struct SetBackgroundColorCommandD
{
  float m_r, m_g, m_b;
};
CMDDECL(SetBackgroundColorCommand, 36, "background_color", CMD_ARGS({ F32, F32, F32 }));
