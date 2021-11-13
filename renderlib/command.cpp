#include "command.h"

#include "AppScene.h"
#include "CCamera.h"
#include "FileReader.h"
#include "ImageXYZC.h"
#include "Logging.h"
#include "RenderSettings.h"
#include "VolumeDimensions.h"

#include "json/json.hpp"

#include <errno.h>
#include <sys/stat.h>

void
SessionCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "Session command: " << m_data.m_name;
}

void
AssetPathCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "AssetPath command: " << m_data.m_name;
}

void
LoadOmeTifCommand::execute(ExecutionContext* c)
{
  LOG_WARNING << "LoadOmeTif command is deprecated. Prefer LoadVolumeFromFile command.";
  LOG_DEBUG << "LoadOmeTif command: " << m_data.m_name;
  struct stat buf;
  if (stat(m_data.m_name.c_str(), &buf) == 0) {
    std::shared_ptr<ImageXYZC> image = FileReader::loadFromFile_4D(m_data.m_name);
    if (!image) {
      return;
    }

    c->m_currentFilePath = m_data.m_name;
    c->m_currentScene = 0;

    c->m_appScene->m_volume = image;
    c->m_appScene->initSceneFromImg(image);

    // Tell the camera about the volume's bounding box
    c->m_camera->m_SceneBoundingBox.m_MinP = c->m_appScene->m_boundingBox.GetMinP();
    c->m_camera->m_SceneBoundingBox.m_MaxP = c->m_appScene->m_boundingBox.GetMaxP();
    c->m_camera->SetViewMode(ViewModeFront);

    // enable up to first three channels!
    // TODO Why should it be three?
    for (uint32_t i = 0; i < image->sizeC(); ++i) {
      c->m_appScene->m_material.m_enabled[i] = (i < 3);
      c->m_appScene->m_material.m_opacity[i] = 1.0f;
    }
    c->m_renderSettings->SetNoIterations(0);
    c->m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
    c->m_renderSettings->m_DirtyFlags.SetFlag(VolumeDirty);
    c->m_renderSettings->m_DirtyFlags.SetFlag(VolumeDataDirty);
    c->m_renderSettings->m_DirtyFlags.SetFlag(TransferFunctionDirty);
    // fire back some json immediately...
    nlohmann::json j;
    j["commandId"] = (int)LoadOmeTifCommand::m_ID;
    j["x"] = (int)image->sizeX();
    j["y"] = (int)image->sizeY();
    j["z"] = (int)image->sizeZ();
    j["c"] = (int)image->sizeC();
    j["t"] = 1;
    j["pixel_size_x"] = image->physicalSizeX();
    j["pixel_size_y"] = image->physicalSizeY();
    j["pixel_size_z"] = image->physicalSizeZ();
    std::vector<std::string> channelNames;
    for (uint32_t i = 0; i < image->sizeC(); ++i) {
      channelNames.push_back((image->channel(i)->m_name));
    }
    j["channel_names"] = channelNames;
    std::vector<uint16_t> channelMaxIntensity;
    for (uint32_t i = 0; i < image->sizeC(); ++i) {
      channelMaxIntensity.push_back(image->channel(i)->m_max);
    }
    j["channel_max_intensity"] = channelMaxIntensity;

    c->m_message = j.dump();
  } else {
    LOG_WARNING << "stat failed on image with errno " << errno;
  }
}

void
SetCameraPosCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetCameraPos " << m_data.m_x << " " << m_data.m_y << " " << m_data.m_z;
  c->m_camera->m_From.x = m_data.m_x;
  c->m_camera->m_From.y = m_data.m_y;
  c->m_camera->m_From.z = m_data.m_z;
  c->m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
}

void
SetCameraTargetCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetCameraTarget " << m_data.m_x << " " << m_data.m_y << " " << m_data.m_z;
  c->m_camera->m_Target.x = m_data.m_x;
  c->m_camera->m_Target.y = m_data.m_y;
  c->m_camera->m_Target.z = m_data.m_z;
  c->m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
}

void
SetCameraUpCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetCameraUp " << m_data.m_x << " " << m_data.m_y << " " << m_data.m_z;
  c->m_camera->m_Up.x = m_data.m_x;
  c->m_camera->m_Up.y = m_data.m_y;
  c->m_camera->m_Up.z = m_data.m_z;
  c->m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
}

void
SetCameraApertureCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetCameraAperture " << m_data.m_x;
  c->m_camera->m_Aperture.m_Size = m_data.m_x;
  c->m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
}
void
SetCameraProjectionCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetCameraProjection " << m_data.m_projectionType << " " << m_data.m_x;
  // We expect to get a 0 for PERSPECTIVE, or 1 for ORTHOGRAPHIC
  // Anything else will be treated as PERSPECTIVE
  c->m_camera->SetProjectionMode(m_data.m_projectionType == 1 ? ORTHOGRAPHIC : PERSPECTIVE);
  if (c->m_camera->m_Projection == ORTHOGRAPHIC) {
    c->m_camera->m_OrthoScale = m_data.m_x;
  } else {
    c->m_camera->m_FovV = m_data.m_x;
  }
  c->m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
}
void
SetCameraFocalDistanceCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetCameraFocalDistance " << m_data.m_x;

  // TODO: how will we ever set the camera back to auto focus?
  c->m_camera->m_Focus.m_Type = Focus::Manual;

  c->m_camera->m_Focus.m_FocalDistance = m_data.m_x;
  c->m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
}
void
SetCameraExposureCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetCameraExposure " << m_data.m_x;
  // 0 is darkness, 1 is max
  c->m_camera->m_Film.m_Exposure = 1.0f - m_data.m_x;
  c->m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
}
void
SetDiffuseColorCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetDiffuse " << m_data.m_channel << " " << m_data.m_r << " " << m_data.m_g << " " << m_data.m_b;
  c->m_appScene->m_material.m_diffuse[m_data.m_channel * 3 + 0] = m_data.m_r;
  c->m_appScene->m_material.m_diffuse[m_data.m_channel * 3 + 1] = m_data.m_g;
  c->m_appScene->m_material.m_diffuse[m_data.m_channel * 3 + 2] = m_data.m_b;
  c->m_renderSettings->SetNoIterations(0);
}
void
SetSpecularColorCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetSpecular " << m_data.m_channel << " " << m_data.m_r << " " << m_data.m_g << " " << m_data.m_b;
  c->m_appScene->m_material.m_specular[m_data.m_channel * 3 + 0] = m_data.m_r;
  c->m_appScene->m_material.m_specular[m_data.m_channel * 3 + 1] = m_data.m_g;
  c->m_appScene->m_material.m_specular[m_data.m_channel * 3 + 2] = m_data.m_b;
  c->m_renderSettings->SetNoIterations(0);
}
void
SetEmissiveColorCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetEmissive " << m_data.m_channel << " " << m_data.m_r << " " << m_data.m_g << " " << m_data.m_b;
  c->m_appScene->m_material.m_emissive[m_data.m_channel * 3 + 0] = m_data.m_r;
  c->m_appScene->m_material.m_emissive[m_data.m_channel * 3 + 1] = m_data.m_g;
  c->m_appScene->m_material.m_emissive[m_data.m_channel * 3 + 2] = m_data.m_b;
  c->m_renderSettings->SetNoIterations(0);
}
void
SetRenderIterationsCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetRenderIterations " << m_data.m_x;
  c->m_camera->m_Film.m_ExposureIterations = m_data.m_x;
}
void
SetStreamModeCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetStreamMode " << m_data.m_x;
  if (c->m_renderer) {
    c->m_renderer->setStreamMode(m_data.m_x);
  }
}
void
RequestRedrawCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "RequestRedraw";
  //	c->_renderer->renderNow();
}
void
SetResolutionCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetResolution " << m_data.m_x << " " << m_data.m_y;
  c->m_camera->m_Film.m_Resolution.SetResX(m_data.m_x);
  c->m_camera->m_Film.m_Resolution.SetResY(m_data.m_y);
  if (c->m_renderer) {
    c->m_renderer->resizeGL(m_data.m_x, m_data.m_y);
  }
  c->m_renderSettings->SetNoIterations(0);
}
void
SetDensityCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetDensity " << m_data.m_x;
  c->m_renderSettings->m_RenderSettings.m_DensityScale = m_data.m_x;
  c->m_renderSettings->SetNoIterations(0);
}

void
FrameSceneCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "FrameScene";
  c->m_camera->m_SceneBoundingBox.m_MinP = c->m_appScene->m_boundingBox.GetMinP();
  c->m_camera->m_SceneBoundingBox.m_MaxP = c->m_appScene->m_boundingBox.GetMaxP();
  c->m_camera->SetViewMode(ViewModeFront);
  c->m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
}
void
SetGlossinessCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetGlossiness " << m_data.m_channel << " " << m_data.m_glossiness;
  c->m_appScene->m_material.m_roughness[m_data.m_channel] = m_data.m_glossiness;
  c->m_renderSettings->m_DirtyFlags.SetFlag(TransferFunctionDirty);
}
void
EnableChannelCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "EnableChannel " << m_data.m_channel << " " << m_data.m_enabled;
  // 0 or 1 hopefully.
  c->m_appScene->m_material.m_enabled[m_data.m_channel] = (m_data.m_enabled != 0);
  c->m_renderSettings->m_DirtyFlags.SetFlag(VolumeDataDirty);
}
void
SetWindowLevelCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetWindowLevel " << m_data.m_channel << " " << m_data.m_window << " " << m_data.m_level;
  GradientData& lutInfo = c->m_appScene->m_material.m_gradientData[m_data.m_channel];
  lutInfo.m_activeMode = GradientEditMode::WINDOW_LEVEL;
  lutInfo.m_window = m_data.m_window;
  lutInfo.m_level = m_data.m_level;
  c->m_appScene->m_volume->channel(m_data.m_channel)->generateFromGradientData(lutInfo);
  c->m_renderSettings->m_DirtyFlags.SetFlag(TransferFunctionDirty);
}
void
OrbitCameraCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "OrbitCamera " << m_data.m_theta << " " << m_data.m_phi;
  c->m_camera->Orbit(m_data.m_theta, m_data.m_phi);
  c->m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
}
void
SetSkylightTopColorCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetSkylightTopColor " << m_data.m_r << " " << m_data.m_g << " " << m_data.m_b;
  c->m_appScene->m_lighting.m_Lights[0].m_ColorTop = glm::vec3(m_data.m_r, m_data.m_g, m_data.m_b);
  c->m_renderSettings->m_DirtyFlags.SetFlag(LightsDirty);
}
void
SetSkylightMiddleColorCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetSkylightMiddleColor " << m_data.m_r << " " << m_data.m_g << " " << m_data.m_b;
  c->m_appScene->m_lighting.m_Lights[0].m_ColorMiddle = glm::vec3(m_data.m_r, m_data.m_g, m_data.m_b);
  c->m_renderSettings->m_DirtyFlags.SetFlag(LightsDirty);
}
void
SetSkylightBottomColorCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetSkylightBottomColor " << m_data.m_r << " " << m_data.m_g << " " << m_data.m_b;
  c->m_appScene->m_lighting.m_Lights[0].m_ColorBottom = glm::vec3(m_data.m_r, m_data.m_g, m_data.m_b);
  c->m_renderSettings->m_DirtyFlags.SetFlag(LightsDirty);
}
void
SetLightPosCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetLightPos " << m_data.m_r << " " << m_data.m_theta << " " << m_data.m_phi;
  c->m_appScene->m_lighting.m_Lights[1 + m_data.m_index].m_Distance = m_data.m_r;
  c->m_appScene->m_lighting.m_Lights[1 + m_data.m_index].m_Theta = m_data.m_theta;
  c->m_appScene->m_lighting.m_Lights[1 + m_data.m_index].m_Phi = m_data.m_phi;
  c->m_renderSettings->m_DirtyFlags.SetFlag(LightsDirty);
}
void
SetLightColorCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetLightColor " << m_data.m_index << " " << m_data.m_r << " " << m_data.m_g << " " << m_data.m_b;
  c->m_appScene->m_lighting.m_Lights[1 + m_data.m_index].m_Color = glm::vec3(m_data.m_r, m_data.m_g, m_data.m_b);
  c->m_renderSettings->m_DirtyFlags.SetFlag(LightsDirty);
}
void
SetLightSizeCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetLightSize " << m_data.m_index << " " << m_data.m_x << " " << m_data.m_y;
  c->m_appScene->m_lighting.m_Lights[1 + m_data.m_index].m_Width = m_data.m_x;
  c->m_appScene->m_lighting.m_Lights[1 + m_data.m_index].m_Height = m_data.m_y;
  c->m_renderSettings->m_DirtyFlags.SetFlag(LightsDirty);
}
void
SetClipRegionCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetClipRegion " << m_data.m_minx << " " << m_data.m_maxx << " " << m_data.m_miny << " " << m_data.m_maxy
            << " " << m_data.m_minz << " " << m_data.m_maxz;
  c->m_appScene->m_roi.SetMinP(glm::vec3(m_data.m_minx, m_data.m_miny, m_data.m_minz));
  c->m_appScene->m_roi.SetMaxP(glm::vec3(m_data.m_maxx, m_data.m_maxy, m_data.m_maxz));
  c->m_renderSettings->m_DirtyFlags.SetFlag(RoiDirty);
}
void
SetVoxelScaleCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetVoxelScale " << m_data.m_x << " " << m_data.m_y << " " << m_data.m_z;
  c->m_appScene->m_volume->setPhysicalSize(m_data.m_x, m_data.m_y, m_data.m_z);
  // update things that depend on this scaling!
  c->m_appScene->initBoundsFromImg(c->m_appScene->m_volume);
  c->m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
}
void
AutoThresholdCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "AutoThreshold " << m_data.m_channel << " " << m_data.m_method;
  GradientData& lutInfo = c->m_appScene->m_material.m_gradientData[m_data.m_channel];

  switch (m_data.m_method) {
    case 0:
      // TODO generate custom control points for a CUSTOM GradientEditMode
      c->m_appScene->m_volume->channel(m_data.m_channel)->generate_auto2();
      break;
    case 1:
      // TODO generate custom control points for a CUSTOM GradientEditMode
      c->m_appScene->m_volume->channel(m_data.m_channel)->generate_auto();
      break;
    case 2:
      // TODO generate custom control points for a CUSTOM GradientEditMode
      c->m_appScene->m_volume->channel(m_data.m_channel)->generate_bestFit();
      break;
    case 3:
      // TODO generate custom control points for a CUSTOM GradientEditMode
      c->m_appScene->m_volume->channel(m_data.m_channel)->generate_chimerax();
      break;
    case 4:
      lutInfo.m_activeMode = GradientEditMode::PERCENTILE;
      lutInfo.m_pctLow = Histogram::DEFAULT_PCT_LOW;
      lutInfo.m_pctHigh = Histogram::DEFAULT_PCT_HIGH;
      c->m_appScene->m_volume->channel(m_data.m_channel)->generateFromGradientData(lutInfo);
      break;
    default:
      LOG_WARNING << "AutoThreshold got unexpected method parameter " << m_data.m_method;
      lutInfo.m_activeMode = GradientEditMode::PERCENTILE;
      lutInfo.m_pctLow = Histogram::DEFAULT_PCT_LOW;
      lutInfo.m_pctHigh = Histogram::DEFAULT_PCT_HIGH;
      c->m_appScene->m_volume->channel(m_data.m_channel)->generateFromGradientData(lutInfo);
      break;
  }
  c->m_renderSettings->m_DirtyFlags.SetFlag(TransferFunctionDirty);
}
void
SetPercentileThresholdCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetPercentileThreshold " << m_data.m_channel << " " << m_data.m_pctLow << " " << m_data.m_pctHigh;
  GradientData& lutInfo = c->m_appScene->m_material.m_gradientData[m_data.m_channel];
  lutInfo.m_activeMode = GradientEditMode::PERCENTILE;
  lutInfo.m_pctLow = m_data.m_pctLow;
  lutInfo.m_pctHigh = m_data.m_pctHigh;
  c->m_appScene->m_volume->channel(m_data.m_channel)->generateFromGradientData(lutInfo);
  c->m_renderSettings->m_DirtyFlags.SetFlag(TransferFunctionDirty);
}
void
SetOpacityCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetOpacity " << m_data.m_channel << " " << m_data.m_opacity;
  c->m_appScene->m_material.m_opacity[m_data.m_channel] = m_data.m_opacity;
  c->m_renderSettings->SetNoIterations(0);
}
void
SetPrimaryRayStepSizeCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetPrimaryRayStepSize " << m_data.m_stepSize;
  c->m_renderSettings->m_RenderSettings.m_StepSizeFactor = m_data.m_stepSize;
  c->m_renderSettings->m_DirtyFlags.SetFlag(RenderParamsDirty);
}
void
SetSecondaryRayStepSizeCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetSecondaryRayStepSize " << m_data.m_stepSize;
  c->m_renderSettings->m_RenderSettings.m_StepSizeFactorShadow = m_data.m_stepSize;
  c->m_renderSettings->m_DirtyFlags.SetFlag(RenderParamsDirty);
}
void
SetBackgroundColorCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetBackgroundColor " << m_data.m_r << ", " << m_data.m_g << ", " << m_data.m_b;
  c->m_appScene->m_material.m_backgroundColor[0] = m_data.m_r;
  c->m_appScene->m_material.m_backgroundColor[1] = m_data.m_g;
  c->m_appScene->m_material.m_backgroundColor[2] = m_data.m_b;
  c->m_renderSettings->m_DirtyFlags.SetFlag(RenderParamsDirty);
}

void
SetIsovalueThresholdCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetIsovalueThreshold " << m_data.m_channel << " " << m_data.m_isovalue << " " << m_data.m_isorange;

  GradientData& lutInfo = c->m_appScene->m_material.m_gradientData[m_data.m_channel];
  lutInfo.m_activeMode = GradientEditMode::ISOVALUE;
  lutInfo.m_isorange = m_data.m_isorange;
  lutInfo.m_isovalue = m_data.m_isovalue;
  c->m_appScene->m_volume->channel(m_data.m_channel)->generateFromGradientData(lutInfo);
  c->m_renderSettings->m_DirtyFlags.SetFlag(TransferFunctionDirty);
}

void
SetControlPointsCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetControlPoints " << m_data.m_channel;
  // TODO debug print the data
  GradientData& lutInfo = c->m_appScene->m_material.m_gradientData[m_data.m_channel];
  lutInfo.m_activeMode = GradientEditMode::CUSTOM;
  std::vector<LutControlPoint> stops;
  // 5 floats per stop.  first is position, next four are rgba.  use a only, for now.
  // TODO SHOULD PARSE DO THIS JOB?
  for (size_t i = 0; i < m_data.m_data.size() / 5; ++i) {
    stops.push_back({ m_data.m_data[i * 5], m_data.m_data[i * 5 + 4] });
  }

  lutInfo.m_customControlPoints = stops;
  c->m_appScene->m_volume->channel(m_data.m_channel)->generateFromGradientData(lutInfo);

  c->m_renderSettings->m_DirtyFlags.SetFlag(TransferFunctionDirty);
}

void
LoadVolumeFromFileCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "LoadVolumeFromFile command: " << m_data.m_path << " S=" << m_data.m_scene << " T=" << m_data.m_time;
  struct stat buf;
  if (stat(m_data.m_path.c_str(), &buf) == 0) {
    VolumeDimensions dims;
    // note T and S args are swapped in order here. this is intentional.
    std::shared_ptr<ImageXYZC> image = FileReader::loadFromFile(m_data.m_path, &dims, m_data.m_time, m_data.m_scene);
    if (!image) {
      return;
    }

    c->m_currentFilePath = m_data.m_path;
    c->m_currentScene = m_data.m_scene;

    c->m_appScene->m_timeLine.setRange(0, dims.sizeT - 1);
    c->m_appScene->m_timeLine.setCurrentTime(m_data.m_time);

    c->m_appScene->m_volume = image;
    c->m_appScene->initSceneFromImg(image);

    // Tell the camera about the volume's bounding box
    c->m_camera->m_SceneBoundingBox.m_MinP = c->m_appScene->m_boundingBox.GetMinP();
    c->m_camera->m_SceneBoundingBox.m_MaxP = c->m_appScene->m_boundingBox.GetMaxP();
    c->m_camera->SetViewMode(ViewModeFront);

    // enable up to first three channels!
    // TODO Why should it be three?
    for (uint32_t i = 0; i < image->sizeC(); ++i) {
      c->m_appScene->m_material.m_enabled[i] = (i < 3);
      c->m_appScene->m_material.m_opacity[i] = 1.0f;
    }
    c->m_renderSettings->SetNoIterations(0);
    c->m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
    c->m_renderSettings->m_DirtyFlags.SetFlag(VolumeDirty);
    c->m_renderSettings->m_DirtyFlags.SetFlag(VolumeDataDirty);
    c->m_renderSettings->m_DirtyFlags.SetFlag(TransferFunctionDirty);

    // fire back some json immediately...
    nlohmann::json j;
    j["commandId"] = (int)LoadVolumeFromFileCommand::m_ID;
    j["x"] = (int)image->sizeX();
    j["y"] = (int)image->sizeY();
    j["z"] = (int)image->sizeZ();
    j["c"] = (int)image->sizeC();
    j["t"] = 1;
    j["pixel_size_x"] = image->physicalSizeX();
    j["pixel_size_y"] = image->physicalSizeY();
    j["pixel_size_z"] = image->physicalSizeZ();
    std::vector<std::string> channelNames;
    for (uint32_t i = 0; i < image->sizeC(); ++i) {
      channelNames.push_back((image->channel(i)->m_name));
    }
    j["channel_names"] = channelNames;
    std::vector<uint16_t> channelMaxIntensity;
    for (uint32_t i = 0; i < image->sizeC(); ++i) {
      channelMaxIntensity.push_back(image->channel(i)->m_max);
    }
    j["channel_max_intensity"] = channelMaxIntensity;

    c->m_message = j.dump();
  } else {
    LOG_WARNING << "stat failed on image with errno " << errno;
  }
}

void
SetTimeCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetTime command: "
            << " T=" << m_data.m_time;

  // setting same time is a no-op.
  if (m_data.m_time == c->m_appScene->m_timeLine.currentTime()) {
    return;
  }

  struct stat buf;
  if (stat(c->m_currentFilePath.c_str(), &buf) == 0) {
    VolumeDimensions dims;
    // note T and S args are swapped in order here. this is intentional.
    std::shared_ptr<ImageXYZC> image =
      FileReader::loadFromFile(c->m_currentFilePath, &dims, m_data.m_time, c->m_currentScene);
    if (!image) {
      return;
    }

    c->m_appScene->m_timeLine.setCurrentTime(m_data.m_time);

    // we expect the scene volume dimensions to be the same; we want to preserve all view settings here.
    // BUT we want to convert the old lookup tables to new lookup tables
    // if we are preserving absolute transfer function settings

    // assume sizeC is same for both previous image and new image!
    if (image->sizeC() != c->m_appScene->m_volume->sizeC()) {
      LOG_ERROR << "Channel count mismatch for different times in same file";
    }

    // remap LUTs to preserve absolute thresholding
    for (uint32_t i = 0; i < image->sizeC(); ++i) {
      GradientData& lutInfo = c->m_appScene->m_material.m_gradientData[i];
      lutInfo.convert(c->m_appScene->m_volume->channel(i)->m_histogram, image->channel(i)->m_histogram);

      image->channel(i)->generateFromGradientData(lutInfo);
    }

    // now we're ready to lose the old channel histograms
    c->m_appScene->m_volume = image;

    c->m_renderSettings->m_DirtyFlags.SetFlag(VolumeDirty);
    c->m_renderSettings->m_DirtyFlags.SetFlag(VolumeDataDirty);
    c->m_renderSettings->m_DirtyFlags.SetFlag(TransferFunctionDirty);

    // fire back some json immediately...
    nlohmann::json j;
    j["commandId"] = (int)SetTimeCommand::m_ID;
    std::vector<uint16_t> channelMaxIntensity;
    for (uint32_t i = 0; i < image->sizeC(); ++i) {
      channelMaxIntensity.push_back(image->channel(i)->m_max);
    }
    j["channel_max_intensity"] = channelMaxIntensity;

    c->m_message = j.dump();

  } else {
    LOG_WARNING << "stat failed on image with errno " << errno;
    LOG_WARNING << "SetTime command called without a file loaded";
  }
}

SessionCommand*
SessionCommand::parse(ParseableStream* c)
{
  SessionCommandD data;
  data.m_name = c->parseString();
  return new SessionCommand(data);
}
AssetPathCommand*
AssetPathCommand::parse(ParseableStream* c)
{
  AssetPathCommandD data;
  data.m_name = c->parseString();
  return new AssetPathCommand(data);
}
LoadOmeTifCommand*
LoadOmeTifCommand::parse(ParseableStream* c)
{
  LOG_WARNING << "LoadOmeTif command is deprecated. Prefer LoadVolumeFromFile command.";
  LoadOmeTifCommandD data;
  data.m_name = c->parseString();
  return new LoadOmeTifCommand(data);
}
SetCameraPosCommand*
SetCameraPosCommand::parse(ParseableStream* c)
{
  SetCameraPosCommandD data;
  data.m_x = c->parseFloat32();
  data.m_y = c->parseFloat32();
  data.m_z = c->parseFloat32();
  return new SetCameraPosCommand(data);
}
SetCameraUpCommand*
SetCameraUpCommand::parse(ParseableStream* c)
{
  SetCameraUpCommandD data;
  data.m_x = c->parseFloat32();
  data.m_y = c->parseFloat32();
  data.m_z = c->parseFloat32();
  return new SetCameraUpCommand(data);
}
SetCameraTargetCommand*
SetCameraTargetCommand::parse(ParseableStream* c)
{
  SetCameraTargetCommandD data;
  data.m_x = c->parseFloat32();
  data.m_y = c->parseFloat32();
  data.m_z = c->parseFloat32();
  return new SetCameraTargetCommand(data);
}
SetCameraApertureCommand*
SetCameraApertureCommand::parse(ParseableStream* c)
{
  SetCameraApertureCommandD data;
  data.m_x = c->parseFloat32();
  return new SetCameraApertureCommand(data);
}
SetCameraProjectionCommand*
SetCameraProjectionCommand::parse(ParseableStream* c)
{
  SetCameraProjectionCommandD data;
  data.m_projectionType = c->parseInt32();
  data.m_x = c->parseFloat32();
  return new SetCameraProjectionCommand(data);
}
SetCameraFocalDistanceCommand*
SetCameraFocalDistanceCommand::parse(ParseableStream* c)
{
  SetCameraFocalDistanceCommandD data;
  data.m_x = c->parseFloat32();
  return new SetCameraFocalDistanceCommand(data);
}
SetCameraExposureCommand*
SetCameraExposureCommand::parse(ParseableStream* c)
{
  SetCameraExposureCommandD data;
  data.m_x = c->parseFloat32();
  return new SetCameraExposureCommand(data);
}
SetDiffuseColorCommand*
SetDiffuseColorCommand::parse(ParseableStream* c)
{
  SetDiffuseColorCommandD data;
  data.m_channel = c->parseInt32();
  data.m_r = c->parseFloat32();
  data.m_g = c->parseFloat32();
  data.m_b = c->parseFloat32();
  data.m_a = c->parseFloat32();
  return new SetDiffuseColorCommand(data);
}
SetSpecularColorCommand*
SetSpecularColorCommand::parse(ParseableStream* c)
{
  SetSpecularColorCommandD data;
  data.m_channel = c->parseInt32();
  data.m_r = c->parseFloat32();
  data.m_g = c->parseFloat32();
  data.m_b = c->parseFloat32();
  data.m_a = c->parseFloat32();
  return new SetSpecularColorCommand(data);
}
SetEmissiveColorCommand*
SetEmissiveColorCommand::parse(ParseableStream* c)
{
  SetEmissiveColorCommandD data;
  data.m_channel = c->parseInt32();
  data.m_r = c->parseFloat32();
  data.m_g = c->parseFloat32();
  data.m_b = c->parseFloat32();
  data.m_a = c->parseFloat32();
  return new SetEmissiveColorCommand(data);
}
SetRenderIterationsCommand*
SetRenderIterationsCommand::parse(ParseableStream* c)
{
  SetRenderIterationsCommandD data;
  data.m_x = c->parseInt32();
  return new SetRenderIterationsCommand(data);
}
SetStreamModeCommand*
SetStreamModeCommand::parse(ParseableStream* c)
{
  SetStreamModeCommandD data;
  data.m_x = c->parseInt32();
  return new SetStreamModeCommand(data);
}
RequestRedrawCommand*
RequestRedrawCommand::parse(ParseableStream* c)
{
  RequestRedrawCommandD data;
  return new RequestRedrawCommand(data);
}
SetResolutionCommand*
SetResolutionCommand::parse(ParseableStream* c)
{
  SetResolutionCommandD data;
  data.m_x = c->parseInt32();
  data.m_y = c->parseInt32();
  return new SetResolutionCommand(data);
}
SetDensityCommand*
SetDensityCommand::parse(ParseableStream* c)
{
  SetDensityCommandD data;
  data.m_x = c->parseFloat32();
  return new SetDensityCommand(data);
}
FrameSceneCommand*
FrameSceneCommand::parse(ParseableStream* c)
{
  FrameSceneCommandD data;
  return new FrameSceneCommand(data);
}
SetGlossinessCommand*
SetGlossinessCommand::parse(ParseableStream* c)
{
  SetGlossinessCommandD data;
  data.m_channel = c->parseInt32();
  data.m_glossiness = c->parseFloat32();
  return new SetGlossinessCommand(data);
}
EnableChannelCommand*
EnableChannelCommand::parse(ParseableStream* c)
{
  EnableChannelCommandD data;
  data.m_channel = c->parseInt32();
  data.m_enabled = c->parseInt32();
  return new EnableChannelCommand(data);
}
SetWindowLevelCommand*
SetWindowLevelCommand::parse(ParseableStream* c)
{
  SetWindowLevelCommandD data;
  data.m_channel = c->parseInt32();
  data.m_window = c->parseFloat32();
  data.m_level = c->parseFloat32();
  return new SetWindowLevelCommand(data);
}
OrbitCameraCommand*
OrbitCameraCommand::parse(ParseableStream* c)
{
  OrbitCameraCommandD data;
  data.m_theta = c->parseFloat32();
  data.m_phi = c->parseFloat32();
  return new OrbitCameraCommand(data);
}
SetSkylightTopColorCommand*
SetSkylightTopColorCommand::parse(ParseableStream* c)
{
  SetSkylightTopColorCommandD data;
  data.m_r = c->parseFloat32();
  data.m_g = c->parseFloat32();
  data.m_b = c->parseFloat32();
  return new SetSkylightTopColorCommand(data);
}
SetSkylightMiddleColorCommand*
SetSkylightMiddleColorCommand::parse(ParseableStream* c)
{
  SetSkylightMiddleColorCommandD data;
  data.m_r = c->parseFloat32();
  data.m_g = c->parseFloat32();
  data.m_b = c->parseFloat32();
  return new SetSkylightMiddleColorCommand(data);
}
SetSkylightBottomColorCommand*
SetSkylightBottomColorCommand::parse(ParseableStream* c)
{
  SetSkylightBottomColorCommandD data;
  data.m_r = c->parseFloat32();
  data.m_g = c->parseFloat32();
  data.m_b = c->parseFloat32();
  return new SetSkylightBottomColorCommand(data);
}

SetLightPosCommand*
SetLightPosCommand::parse(ParseableStream* c)
{
  SetLightPosCommandD data;
  data.m_index = c->parseInt32();
  data.m_r = c->parseFloat32();
  data.m_theta = c->parseFloat32();
  data.m_phi = c->parseFloat32();
  return new SetLightPosCommand(data);
}
SetLightColorCommand*
SetLightColorCommand::parse(ParseableStream* c)
{
  SetLightColorCommandD data;
  data.m_index = c->parseInt32();
  data.m_r = c->parseFloat32();
  data.m_g = c->parseFloat32();
  data.m_b = c->parseFloat32();
  return new SetLightColorCommand(data);
}
SetLightSizeCommand*
SetLightSizeCommand::parse(ParseableStream* c)
{
  SetLightSizeCommandD data;
  data.m_index = c->parseInt32();
  data.m_x = c->parseFloat32();
  data.m_y = c->parseFloat32();
  return new SetLightSizeCommand(data);
}

float
clamp(float x, float bottom, float top)
{
  return std::min(top, std::max(bottom, x));
}

SetClipRegionCommand*
SetClipRegionCommand::parse(ParseableStream* c)
{
  SetClipRegionCommandD data;
  data.m_minx = c->parseFloat32();
  data.m_minx = clamp(data.m_minx, 0.0, 1.0);
  data.m_maxx = c->parseFloat32();
  data.m_maxx = clamp(data.m_maxx, 0.0, 1.0);
  data.m_miny = c->parseFloat32();
  data.m_miny = clamp(data.m_miny, 0.0, 1.0);
  data.m_maxy = c->parseFloat32();
  data.m_maxy = clamp(data.m_maxy, 0.0, 1.0);
  data.m_minz = c->parseFloat32();
  data.m_minz = clamp(data.m_minz, 0.0, 1.0);
  data.m_maxz = c->parseFloat32();
  data.m_maxz = clamp(data.m_maxz, 0.0, 1.0);
  return new SetClipRegionCommand(data);
}

SetVoxelScaleCommand*
SetVoxelScaleCommand::parse(ParseableStream* c)
{
  SetVoxelScaleCommandD data;
  data.m_x = c->parseFloat32();
  data.m_y = c->parseFloat32();
  data.m_z = c->parseFloat32();
  return new SetVoxelScaleCommand(data);
}
AutoThresholdCommand*
AutoThresholdCommand::parse(ParseableStream* c)
{
  AutoThresholdCommandD data;
  data.m_channel = c->parseInt32();
  data.m_method = c->parseInt32();
  return new AutoThresholdCommand(data);
}
SetPercentileThresholdCommand*
SetPercentileThresholdCommand::parse(ParseableStream* c)
{
  SetPercentileThresholdCommandD data;
  data.m_channel = c->parseInt32();
  data.m_pctLow = c->parseFloat32();
  data.m_pctHigh = c->parseFloat32();
  return new SetPercentileThresholdCommand(data);
}
SetOpacityCommand*
SetOpacityCommand::parse(ParseableStream* c)
{
  SetOpacityCommandD data;
  data.m_channel = c->parseInt32();
  data.m_opacity = c->parseFloat32();
  return new SetOpacityCommand(data);
}
SetPrimaryRayStepSizeCommand*
SetPrimaryRayStepSizeCommand::parse(ParseableStream* c)
{
  SetPrimaryRayStepSizeCommandD data;
  data.m_stepSize = c->parseFloat32();
  return new SetPrimaryRayStepSizeCommand(data);
}
SetSecondaryRayStepSizeCommand*
SetSecondaryRayStepSizeCommand::parse(ParseableStream* c)
{
  SetSecondaryRayStepSizeCommandD data;
  data.m_stepSize = c->parseFloat32();
  return new SetSecondaryRayStepSizeCommand(data);
}
SetBackgroundColorCommand*
SetBackgroundColorCommand::parse(ParseableStream* c)
{
  SetBackgroundColorCommandD data;
  data.m_r = c->parseFloat32();
  data.m_g = c->parseFloat32();
  data.m_b = c->parseFloat32();
  return new SetBackgroundColorCommand(data);
}
SetIsovalueThresholdCommand*
SetIsovalueThresholdCommand::parse(ParseableStream* c)
{
  SetIsovalueThresholdCommandD data;
  data.m_channel = c->parseInt32();
  data.m_isovalue = c->parseFloat32();
  data.m_isorange = c->parseFloat32();
  return new SetIsovalueThresholdCommand(data);
}
SetControlPointsCommand*
SetControlPointsCommand::parse(ParseableStream* c)
{
  SetControlPointsCommandD data;
  data.m_channel = c->parseInt32();
  data.m_data = c->parseFloat32Array();
  return new SetControlPointsCommand(data);
}

LoadVolumeFromFileCommand*
LoadVolumeFromFileCommand::parse(ParseableStream* c)
{
  LoadVolumeFromFileCommandD data;
  data.m_path = c->parseString();
  data.m_scene = c->parseInt32();
  data.m_time = c->parseInt32();
  return new LoadVolumeFromFileCommand(data);
}

SetTimeCommand*
SetTimeCommand::parse(ParseableStream* c)
{
  SetTimeCommandD data;
  data.m_time = c->parseInt32();
  return new SetTimeCommand(data);
}

std::string
SessionCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << "\"" << m_data.m_name << "\"";
  ss << ")";
  return ss.str();
}
std::string
AssetPathCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << "\"" << m_data.m_name << "\"";
  ss << ")";
  return ss.str();
}
std::string
LoadOmeTifCommand::toPythonString() const
{
  LOG_WARNING << "LoadOmeTif command is deprecated. Prefer LoadVolumeFromFile command.";
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << "\"" << m_data.m_name << "\"";
  ss << ")";
  return ss.str();
}
std::string
SetCameraPosCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_x << ", " << m_data.m_y << ", " << m_data.m_z;
  ss << ")";
  return ss.str();
}
std::string
SetCameraUpCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_x << ", " << m_data.m_y << ", " << m_data.m_z;
  ss << ")";
  return ss.str();
}
std::string
SetCameraTargetCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_x << ", " << m_data.m_y << ", " << m_data.m_z;
  ss << ")";
  return ss.str();
}
std::string
SetCameraApertureCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_x;
  ss << ")";
  return ss.str();
}
std::string
SetCameraProjectionCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_projectionType << ", " << m_data.m_x;
  ss << ")";
  return ss.str();
}
std::string
SetCameraFocalDistanceCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_x;
  ss << ")";
  return ss.str();
}
std::string
SetCameraExposureCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_x;
  ss << ")";
  return ss.str();
}
std::string
SetDiffuseColorCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_channel << ", " << m_data.m_r << ", " << m_data.m_g << ", " << m_data.m_b << ", " << m_data.m_a;
  ss << ")";
  return ss.str();
}
std::string
SetSpecularColorCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_channel << ", " << m_data.m_r << ", " << m_data.m_g << ", " << m_data.m_b << ", " << m_data.m_a;
  ss << ")";
  return ss.str();
}
std::string
SetEmissiveColorCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_channel << ", " << m_data.m_r << ", " << m_data.m_g << ", " << m_data.m_b << ", " << m_data.m_a;
  ss << ")";
  return ss.str();
}
std::string
SetRenderIterationsCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_x;
  ss << ")";
  return ss.str();
}
std::string
SetStreamModeCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_x;
  ss << ")";
  return ss.str();
}
std::string
RequestRedrawCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << ")";
  return ss.str();
}
std::string
SetResolutionCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_x << ", " << m_data.m_y;
  ss << ")";
  return ss.str();
}
std::string
SetDensityCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_x;
  ss << ")";
  return ss.str();
}
std::string
FrameSceneCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << ")";
  return ss.str();
}
std::string
SetGlossinessCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_channel << ", " << m_data.m_glossiness;
  ss << ")";
  return ss.str();
}
std::string
EnableChannelCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_channel << ", " << m_data.m_enabled;
  ss << ")";
  return ss.str();
}
std::string
SetWindowLevelCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_channel << ", " << m_data.m_window << ", " << m_data.m_level;
  ss << ")";
  return ss.str();
}
std::string
OrbitCameraCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_theta << ", " << m_data.m_phi;
  ss << ")";
  return ss.str();
}
std::string
SetSkylightTopColorCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_r << ", " << m_data.m_g << ", " << m_data.m_b;
  ss << ")";
  return ss.str();
}
std::string
SetSkylightMiddleColorCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_r << ", " << m_data.m_g << ", " << m_data.m_b;
  ss << ")";
  return ss.str();
}
std::string
SetSkylightBottomColorCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_r << ", " << m_data.m_g << ", " << m_data.m_b;
  ss << ")";
  return ss.str();
}

std::string
SetLightPosCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_index << ", " << m_data.m_r << ", " << m_data.m_theta << ", " << m_data.m_phi;
  ss << ")";
  return ss.str();
}
std::string
SetLightColorCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_index << ", " << m_data.m_r << ", " << m_data.m_g << ", " << m_data.m_b;
  ss << ")";
  return ss.str();
}
std::string
SetLightSizeCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_index << ", " << m_data.m_x << ", " << m_data.m_y;
  ss << ")";
  return ss.str();
}

std::string
SetClipRegionCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_minx << ", " << m_data.m_maxx << ", " << m_data.m_miny << ", " << m_data.m_maxy << ", "
     << m_data.m_minz << ", " << m_data.m_maxz;
  ss << ")";
  return ss.str();
}

std::string
SetVoxelScaleCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_x << ", " << m_data.m_y << ", " << m_data.m_z;
  ss << ")";
  return ss.str();
}
std::string
AutoThresholdCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_channel << ", " << m_data.m_method;
  ss << ")";
  return ss.str();
}
std::string
SetPercentileThresholdCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_channel << ", " << m_data.m_pctLow << ", " << m_data.m_pctHigh;
  ss << ")";
  return ss.str();
}
std::string
SetOpacityCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_channel << ", " << m_data.m_opacity;
  ss << ")";
  return ss.str();
}
std::string
SetPrimaryRayStepSizeCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_stepSize;
  ss << ")";
  return ss.str();
}
std::string
SetSecondaryRayStepSizeCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_stepSize;
  ss << ")";
  return ss.str();
}
std::string
SetBackgroundColorCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_r << ", " << m_data.m_g << ", " << m_data.m_b;
  ss << ")";
  return ss.str();
}
std::string
SetIsovalueThresholdCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";
  ss << m_data.m_channel << ", " << m_data.m_isovalue << ", " << m_data.m_isorange;
  ss << ")";
  return ss.str();
}
std::string
SetControlPointsCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";

  ss << m_data.m_channel << ", [";
  // insert comma delimited but no comma after the last entry
  if (!m_data.m_data.empty()) {
    std::copy(m_data.m_data.begin(), std::prev(m_data.m_data.end()), std::ostream_iterator<float>(ss, ", "));
    ss << m_data.m_data.back();
  }
  ss << "]";

  ss << ")";
  return ss.str();
}

std::string
LoadVolumeFromFileCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";

  ss << "\"" << m_data.m_path << "\", ";
  ss << m_data.m_scene << ", " << m_data.m_time;

  ss << ")";
  return ss.str();
}

std::string
SetTimeCommand::toPythonString() const
{
  std::ostringstream ss;
  ss << PythonName() << "(";

  ss << m_data.m_time;

  ss << ")";
  return ss.str();
}
