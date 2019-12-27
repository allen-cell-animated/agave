#pragma once

#include <QString>
#include <string>

class CCamera;
class Renderer;
class RenderSettings;
class Scene;

struct ExecutionContext
{
  Renderer* m_renderer;
  RenderSettings* m_renderSettings;
  Scene* m_appScene;
  CCamera* m_camera;
  QString m_message;
};

class Command
{
public:
  // return number of bytes advanced
  virtual void execute(ExecutionContext* context) = 0;

  virtual ~Command() {}
};

#define CMDDECL(NAME, CMDID)                                                                                           \
  class NAME : public Command                                                                                          \
  {                                                                                                                    \
  public:                                                                                                              \
    NAME(NAME##D d)                                                                                                    \
      : m_data(d)                                                                                                      \
    {}                                                                                                                 \
    virtual void execute(ExecutionContext* context);                                                                   \
    static const uint32_t m_ID = CMDID;                                                                                \
    NAME##D m_data;                                                                                                    \
  };

struct SessionCommandD
{
  std::string m_name;
};
CMDDECL(SessionCommand, 0);

struct AssetPathCommandD
{
  std::string m_name;
};
CMDDECL(AssetPathCommand, 1);

struct LoadOmeTifCommandD
{
  std::string m_name;
};
CMDDECL(LoadOmeTifCommand, 2);

struct SetCameraPosCommandD
{
  float m_x, m_y, m_z;
};
CMDDECL(SetCameraPosCommand, 3);

struct SetCameraTargetCommandD
{
  float m_x, m_y, m_z;
};
CMDDECL(SetCameraTargetCommand, 4);

struct SetCameraUpCommandD
{
  float m_x, m_y, m_z;
};
CMDDECL(SetCameraUpCommand, 5);

struct SetCameraApertureCommandD
{
  float m_x;
};
CMDDECL(SetCameraApertureCommand, 6);

struct SetCameraProjectionCommandD
{
  // perspective or ortho
  int32_t m_projectionType;
  // fov degrees or ortho scale
  float m_x;
};
CMDDECL(SetCameraProjectionCommand, 7);

struct SetCameraFocalDistanceCommandD
{
  float m_x;
};
CMDDECL(SetCameraFocalDistanceCommand, 8);

struct SetCameraExposureCommandD
{
  float m_x;
};
CMDDECL(SetCameraExposureCommand, 9);

struct SetDiffuseColorCommandD
{
  int32_t m_channel;
  float m_r, m_g, m_b, m_a;
};
CMDDECL(SetDiffuseColorCommand, 10);

struct SetSpecularColorCommandD
{
  int32_t m_channel;
  float m_r, m_g, m_b, m_a;
};
CMDDECL(SetSpecularColorCommand, 11);

struct SetEmissiveColorCommandD
{
  int32_t m_channel;
  float m_r, m_g, m_b, m_a;
};
CMDDECL(SetEmissiveColorCommand, 12);

struct SetRenderIterationsCommandD
{
  int32_t m_x;
};
CMDDECL(SetRenderIterationsCommand, 13);

struct SetStreamModeCommandD
{
  int32_t m_x;
};
CMDDECL(SetStreamModeCommand, 14);

struct RequestRedrawCommandD
{};
CMDDECL(RequestRedrawCommand, 15);

struct SetResolutionCommandD
{
  int32_t m_x, m_y;
};
CMDDECL(SetResolutionCommand, 16);

struct SetDensityCommandD
{
  float m_x;
};
CMDDECL(SetDensityCommand, 17);

struct FrameSceneCommandD
{};
CMDDECL(FrameSceneCommand, 18);

struct SetGlossinessCommandD
{
  int32_t m_channel;
  float m_glossiness;
};
CMDDECL(SetGlossinessCommand, 19);

struct EnableChannelCommandD
{
  int32_t m_channel;
  int32_t m_enabled;
};
CMDDECL(EnableChannelCommand, 20);

struct SetWindowLevelCommandD
{
  int32_t m_channel;
  float m_window;
  float m_level;
};
CMDDECL(SetWindowLevelCommand, 21);

struct OrbitCameraCommandD
{
  float m_theta;
  float m_phi;
};
CMDDECL(OrbitCameraCommand, 22);

struct SetSkylightTopColorCommandD
{
  float m_r, m_g, m_b;
};
CMDDECL(SetSkylightTopColorCommand, 23);

struct SetSkylightMiddleColorCommandD
{
  float m_r, m_g, m_b;
};
CMDDECL(SetSkylightMiddleColorCommand, 24);

struct SetSkylightBottomColorCommandD
{
  float m_r, m_g, m_b;
};
CMDDECL(SetSkylightBottomColorCommand, 25);

struct SetLightPosCommandD
{
  int32_t m_index;
  float m_r, m_theta, m_phi;
};
CMDDECL(SetLightPosCommand, 26);

struct SetLightColorCommandD
{
  int32_t m_index;
  float m_r, m_g, m_b;
};
CMDDECL(SetLightColorCommand, 27);

struct SetLightSizeCommandD
{
  int32_t m_index;
  float m_x, m_y;
};
CMDDECL(SetLightSizeCommand, 28);

struct SetClipRegionCommandD
{
  float m_minx, m_maxx;
  float m_miny, m_maxy;
  float m_minz, m_maxz;
};
CMDDECL(SetClipRegionCommand, 29);

struct SetVoxelScaleCommandD
{
  float m_x, m_y, m_z;
};
CMDDECL(SetVoxelScaleCommand, 30);

struct AutoThresholdCommandD
{
  int32_t m_channel;
  int32_t m_method;
};
CMDDECL(AutoThresholdCommand, 31);

struct SetPercentileThresholdCommandD
{
  int32_t m_channel;
  float m_pctLow;
  float m_pctHigh;
};
CMDDECL(SetPercentileThresholdCommand, 32);

struct SetOpacityCommandD
{
  int32_t m_channel;
  float m_opacity;
};
CMDDECL(SetOpacityCommand, 33);

struct SetPrimaryRayStepSizeCommandD
{
  float m_stepSize;
};
CMDDECL(SetPrimaryRayStepSizeCommand, 34);

struct SetSecondaryRayStepSizeCommandD
{
  float m_stepSize;
};
CMDDECL(SetSecondaryRayStepSizeCommand, 35);

struct SetBackgroundColorCommandD
{
  float m_r, m_g, m_b;
};
CMDDECL(SetBackgroundColorCommand, 36);
