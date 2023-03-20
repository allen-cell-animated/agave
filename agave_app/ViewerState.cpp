#include "ViewerState.h"

#include "command.h"

#include "renderlib/AppScene.h"
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

std::map<GradientEditMode, Serialize::GradientEditMode_PID> g_GradientModeToPermId = {
  { GradientEditMode::WINDOW_LEVEL, Serialize::GradientEditMode_PID::WINDOW_LEVEL },
  { GradientEditMode::ISOVALUE, Serialize::GradientEditMode_PID::ISOVALUE },
  { GradientEditMode::PERCENTILE, Serialize::GradientEditMode_PID::PERCENTILE },
  { GradientEditMode::CUSTOM, Serialize::GradientEditMode_PID::CUSTOM }
};
std::map<Serialize::GradientEditMode_PID, GradientEditMode> g_PermIdToGradientMode = {
  { Serialize::GradientEditMode_PID::WINDOW_LEVEL, GradientEditMode::WINDOW_LEVEL },
  { Serialize::GradientEditMode_PID::ISOVALUE, GradientEditMode::ISOVALUE },
  { Serialize::GradientEditMode_PID::PERCENTILE, GradientEditMode::PERCENTILE },
  { Serialize::GradientEditMode_PID::CUSTOM, GradientEditMode::CUSTOM }
};

Serialize::ViewerState
stateFromJson(const nlohmann::json& jsonDoc)
{
  // VERSION MUST EXIST.  THROW OR PANIC IF NOT.
  glm::ivec3 v(0, 0, 0);
  getVec3i(jsonDoc, "version", v);
  Version version(v);

  // we will fill this in from the jsonDoc.
  Serialize::ViewerState stateV2;

  // version checks.  Parse old data structures here.
  if (version <= Version(1, 4, 1)) {
    Serialize::ViewerState_V1 stateV1 = jsonDoc.get<Serialize::ViewerState_V1>();
    // fill in this from the old data structure.
    stateV2.fromV1(stateV1);
  } else {
    stateV2 = jsonDoc.get<Serialize::ViewerState>();
  }
  return stateV2;
}

QString
stateToPythonScript(const Serialize::ViewerState& s)
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
            s.camera.projection == Serialize::Projection_PID::PERSPECTIVE ? s.camera.fovY : s.camera.orthoScale })
          .toPythonString()
     << std::endl;

  ss << obj << SetCameraExposureCommand({ s.camera.exposure }).toPythonString() << std::endl;
  ss << obj << SetDensityCommand({ s.density }).toPythonString() << std::endl;
  ss << obj << SetCameraApertureCommand({ s.camera.aperture }).toPythonString() << std::endl;
  ss << obj << SetCameraFocalDistanceCommand({ s.camera.focalDistance }).toPythonString() << std::endl;

  // per-channel
  for (std::int32_t i = 0; i < s.channels.size(); ++i) {
    const Serialize::ChannelSettings_V1& ch = s.channels[i];
    ss << obj << EnableChannelCommand({ i, ch.enabled ? 1 : 0 }).toPythonString() << std::endl;
    ss << obj
       << SetDiffuseColorCommand({ i, ch.diffuseColor[0], ch.diffuseColor[1], ch.diffuseColor[2], 1.0f })
            .toPythonString()
       << std::endl;
    ss << obj
       << SetSpecularColorCommand({ i, ch.specularColor[0], ch.specularColor[1], ch.specularColor[2], 0.0f })
            .toPythonString()
       << std::endl;
    ss << obj
       << SetEmissiveColorCommand({ i, ch.emissiveColor[0], ch.emissiveColor[1], ch.emissiveColor[2], 0.0f })
            .toPythonString()
       << std::endl;
    ss << obj << SetGlossinessCommand({ i, ch.glossiness }).toPythonString() << std::endl;
    ss << obj << SetOpacityCommand({ i, ch.opacity }).toPythonString() << std::endl;
    // depending on current mode:
    switch (g_PermIdToGradientMode[ch.lutParams.mode]) {
      case GradientEditMode::WINDOW_LEVEL:
        ss << obj << SetWindowLevelCommand({ i, ch.lutParams.window, ch.lutParams.level }).toPythonString()
           << std::endl;
        break;
      case GradientEditMode::ISOVALUE:
        ss << obj << SetIsovalueThresholdCommand({ i, ch.lutParams.isovalue, ch.lutParams.isorange }).toPythonString()
           << std::endl;
        break;
      case GradientEditMode::PERCENTILE:
        ss << obj << SetPercentileThresholdCommand({ i, ch.lutParams.pctLow, ch.lutParams.pctHigh }).toPythonString()
           << std::endl;
        break;
      case GradientEditMode::CUSTOM:
        std::vector<float> v;
        for (auto p : ch.lutParams.controlPoints) {
          v.push_back(p.x);
          v.push_back(p.value[0]);
          v.push_back(p.value[1]);
          v.push_back(p.value[2]);
          v.push_back(p.value[3]);
        }
        ss << obj << SetControlPointsCommand({ i, v }).toPythonString() << std::endl;
        break;
    }
  }

  // lighting

  // TODO get sky light.
  // assuming this is light 0 for now.
  const Serialize::LightSettings_V1& light0 = s.lights[0];
  const Serialize::LightSettings_V1& light1 = s.lights[1];
  ss << obj
     << SetSkylightTopColorCommand({ light0.topColor[0] * light0.topColorIntensity,
                                     light0.topColor[1] * light0.topColorIntensity,
                                     light0.topColor[2] * light0.topColorIntensity })
          .toPythonString()
     << std::endl;
  ss << obj
     << SetSkylightMiddleColorCommand({ light0.middleColor[0] * light0.middleColorIntensity,
                                        light0.middleColor[1] * light0.middleColorIntensity,
                                        light0.middleColor[2] * light0.middleColorIntensity })
          .toPythonString()
     << std::endl;
  ss << obj
     << SetSkylightBottomColorCommand({ light0.bottomColor[0] * light0.bottomColorIntensity,
                                        light0.bottomColor[1] * light0.bottomColorIntensity,
                                        light0.bottomColor[2] * light0.bottomColorIntensity })
          .toPythonString()
     << std::endl;
  ss << obj << SetLightPosCommand({ 0, light1.distance, light1.theta, light1.phi }).toPythonString() << std::endl;
  ss << obj
     << SetLightColorCommand({ 0,
                               light1.color[0] * light1.colorIntensity,
                               light1.color[1] * light1.colorIntensity,
                               light1.color[2] * light1.colorIntensity })
          .toPythonString()
     << std::endl;
  ss << obj << SetLightSizeCommand({ 0, light1.width, light1.height }).toPythonString() << std::endl;

  ss << obj << SessionCommand({ outFileName.toStdString() + ".png" }).toPythonString() << std::endl;
  ss << obj << RequestRedrawCommand({}).toPythonString() << std::endl;
  std::string sout(ss.str());
  // LOG_DEBUG << s;
  return QString::fromStdString(sout);
}

LoadSpec
stateToLoadSpec(const Serialize::ViewerState& state)
{
  const Serialize::LoadSettings& s = state.datasets[0];
  LoadSpec spec;
  spec.filepath = s.url;
  spec.subpath = s.subpath;
  spec.scene = s.scene;
  spec.time = s.time;
  spec.channels = s.channels;
  spec.minx = s.clipRegion[0][0];
  spec.maxx = s.clipRegion[0][1];
  spec.miny = s.clipRegion[1][0];
  spec.maxy = s.clipRegion[1][1];
  spec.minz = s.clipRegion[2][0];
  spec.maxz = s.clipRegion[2][1];
  return spec;
}

Serialize::LoadSettings
fromLoadSpec(const LoadSpec& loadSpec)
{
  Serialize::LoadSettings s;
  s.url = loadSpec.filepath;
  s.subpath = loadSpec.subpath;
  s.scene = loadSpec.scene;
  s.time = loadSpec.time;
  s.channels = loadSpec.channels;
  s.clipRegion[0][0] = loadSpec.minx;
  s.clipRegion[0][1] = loadSpec.maxx;
  s.clipRegion[1][0] = loadSpec.miny;
  s.clipRegion[1][1] = loadSpec.maxy;
  s.clipRegion[2][0] = loadSpec.minz;
  s.clipRegion[2][1] = loadSpec.maxz;
  return s;
}

Serialize::LightSettings_V1
fromLight(const Light& lt)
{
  Serialize::LightSettings_V1 s;
  s.type = lt.m_T;
  s.distance = lt.m_Distance;
  s.theta = lt.m_Theta;
  s.phi = lt.m_Phi;
  s.topColor = { lt.m_ColorTop.r, lt.m_ColorTop.g, lt.m_ColorTop.b };
  s.middleColor = { lt.m_ColorMiddle.r, lt.m_ColorMiddle.g, lt.m_ColorMiddle.b };
  s.color = { lt.m_Color.r, lt.m_Color.g, lt.m_Color.b };
  s.bottomColor = { lt.m_ColorBottom.r, lt.m_ColorBottom.g, lt.m_ColorBottom.b };
  s.topColorIntensity = lt.m_ColorTopIntensity;
  s.middleColorIntensity = lt.m_ColorMiddleIntensity;
  s.colorIntensity = lt.m_ColorIntensity;
  s.bottomColorIntensity = lt.m_ColorBottomIntensity;
  s.width = lt.m_Width;
  s.height = lt.m_Height;
  return s;
}