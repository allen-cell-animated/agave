#include "command.h"

#include "renderer.h"

#include "renderlib/AppScene.h"
#include "renderlib/CCamera.h"
#include "renderlib/FileReader.h"
#include "renderlib/ImageXYZC.h"
#include "renderlib/Logging.h"
#include "renderlib/RenderSettings.h"

#include <QElapsedTimer>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

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
  LOG_DEBUG << "LoadOmeTif command: " << m_data.m_name;
  QFileInfo info(QString(m_data.m_name.c_str()));
  if (info.exists()) {
    std::shared_ptr<ImageXYZC> image = FileReader::loadOMETiff_4D(m_data.m_name);
    if (!image) {
      return;
    }

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
    QJsonObject j;
    j["commandId"] = (int)LoadOmeTifCommand::m_ID;
    j["x"] = (int)image->sizeX();
    j["y"] = (int)image->sizeY();
    j["z"] = (int)image->sizeZ();
    j["c"] = (int)image->sizeC();
    j["t"] = 1;
    j["pixel_size_x"] = image->physicalSizeX();
    j["pixel_size_y"] = image->physicalSizeY();
    j["pixel_size_z"] = image->physicalSizeZ();
    QJsonArray channelNames;
    for (uint32_t i = 0; i < image->sizeC(); ++i) {
      channelNames.append(image->channel(i)->m_name);
    }
    j["channel_names"] = channelNames;
    QJsonArray channelMaxIntensity;
    for (uint32_t i = 0; i < image->sizeC(); ++i) {
      channelMaxIntensity.append(image->channel(i)->m_max);
    }
    j["channel_max_intensity"] = channelMaxIntensity;

    QJsonDocument doc(j);
    c->m_message = doc.toJson();
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
  c->m_renderer->setStreamMode(m_data.m_x);
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
  c->m_renderer->resizeGL(m_data.m_x, m_data.m_y);
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
  c->m_appScene->m_volume->channel(m_data.m_channel)->generate_windowLevel(m_data.m_window, m_data.m_level);
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
  c->m_appScene->initSceneFromImg(c->m_appScene->m_volume);
  c->m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
}
void
AutoThresholdCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "AutoThreshold " << m_data.m_channel << " " << m_data.m_method;
  float window, level;
  switch (m_data.m_method) {
    case 0:
      c->m_appScene->m_volume->channel(m_data.m_channel)->generate_auto2(window, level);
      break;
    case 1:
      c->m_appScene->m_volume->channel(m_data.m_channel)->generate_auto(window, level);
      break;
    case 2:
      c->m_appScene->m_volume->channel(m_data.m_channel)->generate_bestFit(window, level);
      break;
    case 3:
      c->m_appScene->m_volume->channel(m_data.m_channel)->generate_chimerax();
      break;
    case 4:
      c->m_appScene->m_volume->channel(m_data.m_channel)->generate_percentiles(window, level);
      break;
    default:
      LOG_WARNING << "AutoThreshold got unexpected method parameter " << m_data.m_method;
      c->m_appScene->m_volume->channel(m_data.m_channel)->generate_percentiles(window, level);
      break;
  }
  c->m_renderSettings->m_DirtyFlags.SetFlag(TransferFunctionDirty);
}
void
SetPercentileThresholdCommand::execute(ExecutionContext* c)
{
  LOG_DEBUG << "SetPercentileThreshold " << m_data.m_channel << " " << m_data.m_pctLow << " " << m_data.m_pctHigh;
  float window, level;
  c->m_appScene->m_volume->channel(m_data.m_channel)
    ->generate_percentiles(window, level, m_data.m_pctLow, m_data.m_pctHigh);
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
