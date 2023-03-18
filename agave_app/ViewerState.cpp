#include "ViewerState.h"

#include "command.h"
#include "renderlib/GradientData.h"
#include "renderlib/Logging.h"
#include "renderlib/version.h"
#include "renderlib/version.hpp"

#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonObject>

#include <sstream>

QJsonArray
jsonVec4(float x, float y, float z, float w)
{
  QJsonArray tgt;
  tgt.append(x);
  tgt.append(y);
  tgt.append(z);
  tgt.append(w);
  return tgt;
}
QJsonArray
jsonVec3(float x, float y, float z)
{
  QJsonArray tgt;
  tgt.append(x);
  tgt.append(y);
  tgt.append(z);
  return tgt;
}
QJsonArray
jsonVec2(float x, float y)
{
  QJsonArray tgt;
  tgt.append(x);
  tgt.append(y);
  return tgt;
}
QJsonArray
jsonVec3(const glm::vec3& v)
{
  QJsonArray tgt;
  tgt.append(v.x);
  tgt.append(v.y);
  tgt.append(v.z);
  return tgt;
}
QJsonArray
jsonVec2(const glm::vec2& v)
{
  QJsonArray tgt;
  tgt.append(v.x);
  tgt.append(v.y);
  return tgt;
}

void
getFloat(const nlohmann::json& obj, std::string prop, float& value)
{
  if (obj.contains(prop)) {
    value = (float)obj[prop].get<double>();
  }
}
void
getInt(const nlohmann::json& obj, std::string prop, int& value)
{
  if (obj.contains(prop)) {
    value = obj[prop].get<int>();
  }
}
void
getString(const nlohmann::json& obj, std::string prop, std::string& value)
{
  if (obj.contains(prop)) {
    value = obj[prop].get<std::string>();
  }
}
void
getBool(const nlohmann::json& obj, std::string prop, bool& value)
{
  if (obj.contains(prop)) {
    value = obj[prop].get<bool>();
  }
}
void
getVec4(const nlohmann::json& obj, std::string prop, glm::vec4& value)
{
  if (obj.contains(prop)) {
    auto ja = obj[prop];
    value.x = ja.at(0).get<double>();
    value.y = ja.at(1).get<double>();
    value.z = ja.at(2).get<double>();
    value.w = ja.at(3).get<double>();
  }
}
void
getVec3(const nlohmann::json& obj, std::string prop, glm::vec3& value)
{
  if (obj.contains(prop)) {
    auto ja = obj[prop];
    value.x = ja.at(0).get<double>();
    value.y = ja.at(1).get<double>();
    value.z = ja.at(2).get<double>();
  }
}
void
getVec2(const nlohmann::json& obj, std::string prop, glm::vec2& value)
{
  if (obj.contains(prop)) {
    auto ja = obj[prop];
    value.x = ja.at(0).get<double>();
    value.y = ja.at(1).get<double>();
  }
}
void
getVec2i(const nlohmann::json& obj, std::string prop, glm::ivec2& value)
{
  if (obj.contains(prop)) {
    auto ja = obj[prop];
    value.x = ja.at(0).get<int32_t>();
    value.y = ja.at(1).get<int32_t>();
  }
}
void
getVec3i(const nlohmann::json& obj, std::string prop, glm::ivec3& value)
{
  if (obj.contains(prop)) {
    auto ja = obj[prop];
    value.x = ja.at(0).get<int32_t>();
    value.y = ja.at(1).get<int32_t>();
    value.z = ja.at(2).get<int32_t>();
  }
}

std::map<GradientEditMode, GradientEditMode_PID> g_GradientModeToPermId = {
  { GradientEditMode::WINDOW_LEVEL, GradientEditMode_PID::WINDOW_LEVEL },
  { GradientEditMode::ISOVALUE, GradientEditMode_PID::ISOVALUE },
  { GradientEditMode::PERCENTILE, GradientEditMode_PID::PERCENTILE },
  { GradientEditMode::CUSTOM, GradientEditMode_PID::CUSTOM }
};
std::map<GradientEditMode_PID, GradientEditMode> g_PermIdToGradientMode = {
  { GradientEditMode_PID::WINDOW_LEVEL, GradientEditMode::WINDOW_LEVEL },
  { GradientEditMode_PID::ISOVALUE, GradientEditMode::ISOVALUE },
  { GradientEditMode_PID::PERCENTILE, GradientEditMode::PERCENTILE },
  { GradientEditMode_PID::CUSTOM, GradientEditMode::CUSTOM }
};

ViewerState
stateFromJson(const nlohmann::json& jsonDoc)
{
  // VERSION MUST EXIST.  THROW OR PANIC IF NOT.
  glm::ivec3 v(0, 0, 0);
  getVec3i(jsonDoc, "version", v);
  Version version(v);

  // we will fill this in from the jsonDoc.
  ViewerState stateV2;

  // version checks.  Parse old data structures here.
  if (version <= Version(1, 4, 1)) {
    ViewerState_V1 stateV1 = jsonDoc.get<ViewerState_V1>();
    // fill in this from the old data structure.
    stateV2.fromV1(stateV1);
  } else {
    stateV2 = jsonDoc.get<ViewerState>();
  }
  return stateV2;
}

QString
stateToPythonScript(const ViewerState& s)
{
  QFileInfo fi(QString::fromStdString(s.datasets[0].url));
  QString outFileName = fi.baseName();

  std::ostringstream ss;
  ss << "# pip install agave_pyclient" << std::endl;
  ss << "# agave --server &" << std::endl;
  ss << "# python myscript.py" << std::endl << std::endl;
  ss << "import agave_pyclient as agave" << std::endl << std::endl;
  ss << "r = agave.AgaveRenderer()" << std::endl;
  std::string obj = "r.";
  ss << obj
     << LoadVolumeFromFileCommand({ s.datasets[0].url, (int32_t)s.datasets[0].scene, (int32_t)s.datasets[0].time })
          .toPythonString()
     << std::endl;
  // TODO use window size or render window capture dims?
  ss << obj << SetResolutionCommand({ s.capture.width, s.capture.height }).toPythonString() << std::endl;
  ss << obj
     << SetBackgroundColorCommand({ s.backgroundColor[0], s.backgroundColor[1], s.backgroundColor[2] }).toPythonString()
     << std::endl;
  ss << obj << ShowBoundingBoxCommand({ s.showBoundingBox }).toPythonString() << std::endl;
  ss << obj
     << SetBoundingBoxColorCommand({ s.boundingBoxColor[0], s.boundingBoxColor[1], s.boundingBoxColor[2] })
          .toPythonString()
     << std::endl;
  // TODO use value from viewport or render window capture settings?
  ss << obj << SetRenderIterationsCommand({ s.capture.samples }).toPythonString() << std::endl;
  ss << obj << SetPrimaryRayStepSizeCommand({ s.pathTracer.primaryStepSize }).toPythonString() << std::endl;
  ss << obj << SetSecondaryRayStepSizeCommand({ s.pathTracer.secondaryStepSize }).toPythonString() << std::endl;
  ss << obj << SetVoxelScaleCommand({ s.scale[0], s.scale[1], s.scale[2] }).toPythonString() << std::endl;
  ss << obj
     << SetClipRegionCommand({ s.clipRegion[0][0],
                               s.clipRegion[0][1],
                               s.clipRegion[1][0],
                               s.clipRegion[1][1],
                               s.clipRegion[2][0],
                               s.clipRegion[2][1] })
          .toPythonString()
     << std::endl;
  ss << obj << SetCameraPosCommand({ s.camera.eye[0], s.camera.eye[1], s.camera.eye[2] }).toPythonString() << std::endl;
  ss << obj << SetCameraTargetCommand({ s.camera.target[0], s.camera.target[1], s.camera.target[2] }).toPythonString()
     << std::endl;
  ss << obj << SetCameraUpCommand({ s.camera.up[0], s.camera.up[1], s.camera.up[2] }).toPythonString() << std::endl;
  ss << obj
     << SetCameraProjectionCommand(
          { s.camera.projection,
            s.camera.projection == Projection_PID::PERSPECTIVE ? s.camera.fovY : s.camera.orthoScale })
          .toPythonString()
     << std::endl;

  ss << obj << SetCameraExposureCommand({ s.camera.exposure }).toPythonString() << std::endl;
  ss << obj << SetDensityCommand({ s.density }).toPythonString() << std::endl;
  ss << obj << SetCameraApertureCommand({ s.camera.aperture }).toPythonString() << std::endl;
  ss << obj << SetCameraFocalDistanceCommand({ s.camera.focalDistance }).toPythonString() << std::endl;

  // per-channel
  for (std::int32_t i = 0; i < s.channels.size(); ++i) {
    const ChannelSettings_V1& ch = s.channels[i];
    ss << obj << EnableChannelCommand({ i, ch.m_enabled ? 1 : 0 }).toPythonString() << std::endl;
    ss << obj << SetDiffuseColorCommand({ i, ch.m_diffuse.x, ch.m_diffuse.y, ch.m_diffuse.z, 1.0f }).toPythonString()
       << std::endl;
    ss << obj
       << SetSpecularColorCommand({ i, ch.m_specular.x, ch.m_specular.y, ch.m_specular.z, 0.0f }).toPythonString()
       << std::endl;
    ss << obj
       << SetEmissiveColorCommand({ i, ch.m_emissive.x, ch.m_emissive.y, ch.m_emissive.z, 0.0f }).toPythonString()
       << std::endl;
    ss << obj << SetGlossinessCommand({ i, ch.m_glossiness }).toPythonString() << std::endl;
    ss << obj << SetOpacityCommand({ i, ch.m_opacity }).toPythonString() << std::endl;
    // depending on current mode:
    switch (LutParams::g_PermIdToGradientMode[ch.m_lutParams.m_mode]) {
      case GradientEditMode::WINDOW_LEVEL:
        ss << obj << SetWindowLevelCommand({ i, ch.m_lutParams.m_window, ch.m_lutParams.m_level }).toPythonString()
           << std::endl;
        break;
      case GradientEditMode::ISOVALUE:
        ss << obj
           << SetIsovalueThresholdCommand({ i, ch.m_lutParams.m_isovalue, ch.m_lutParams.m_isorange }).toPythonString()
           << std::endl;
        break;
      case GradientEditMode::PERCENTILE:
        ss << obj
           << SetPercentileThresholdCommand({ i, ch.m_lutParams.m_pctLow, ch.m_lutParams.m_pctHigh }).toPythonString()
           << std::endl;
        break;
      case GradientEditMode::CUSTOM:
        std::vector<float> v;
        for (auto p : ch.m_lutParams.m_customControlPoints) {
          v.push_back(p.first);
          v.push_back(p.second);
          v.push_back(p.second);
          v.push_back(p.second);
          v.push_back(p.second);
        }
        ss << obj << SetControlPointsCommand({ i, v }).toPythonString() << std::endl;
        break;
    }
  }

  // lighting
  ss << obj
     << SetSkylightTopColorCommand({ m_light0.m_topColor.r * m_light0.m_topColorIntensity,
                                     m_light0.m_topColor.g * m_light0.m_topColorIntensity,
                                     m_light0.m_topColor.b * m_light0.m_topColorIntensity })
          .toPythonString()
     << std::endl;
  ss << obj
     << SetSkylightMiddleColorCommand({ m_light0.m_middleColor.r * m_light0.m_middleColorIntensity,
                                        m_light0.m_middleColor.g * m_light0.m_middleColorIntensity,
                                        m_light0.m_middleColor.b * m_light0.m_middleColorIntensity })
          .toPythonString()
     << std::endl;
  ss << obj
     << SetSkylightBottomColorCommand({ m_light0.m_bottomColor.r * m_light0.m_bottomColorIntensity,
                                        m_light0.m_bottomColor.g * m_light0.m_bottomColorIntensity,
                                        m_light0.m_bottomColor.b * m_light0.m_bottomColorIntensity })
          .toPythonString()
     << std::endl;
  ss << obj << SetLightPosCommand({ 0, m_light1.m_distance, m_light1.m_theta, m_light1.m_phi }).toPythonString()
     << std::endl;
  ss << obj
     << SetLightColorCommand({ 0,
                               m_light1.m_color.r * m_light1.m_colorIntensity,
                               m_light1.m_color.g * m_light1.m_colorIntensity,
                               m_light1.m_color.b * m_light1.m_colorIntensity })
          .toPythonString()
     << std::endl;
  ss << obj << SetLightSizeCommand({ 0, m_light1.m_width, m_light1.m_height }).toPythonString() << std::endl;

  ss << obj << SessionCommand({ outFileName.toStdString() + ".png" }).toPythonString() << std::endl;
  ss << obj << RequestRedrawCommand({}).toPythonString() << std::endl;
  std::string s(ss.str());
  // LOG_DEBUG << s;
  return QString::fromStdString(s);
}
