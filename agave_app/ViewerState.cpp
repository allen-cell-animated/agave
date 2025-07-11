#include "ViewerState.h"

#include "command.h"

#include "renderlib/AppScene.h"
#include "renderlib/CCamera.h"
#include "renderlib/GradientData.h"
#include "renderlib/Logging.h"
#include "renderlib/StringUtil.h"
#include "renderlib/renderlib.h"
#include "renderlib/version.h"
#include "renderlib/version.hpp"

#include <QFile>
#include <QFileInfo>

#include <sstream>
#include <regex>

template<typename K, typename V>
std::unordered_map<V, K>
inverse_map(std::unordered_map<K, V>& map)
{
  std::unordered_map<V, K> inv;
  std::for_each(
    map.begin(), map.end(), [&inv](const std::pair<K, V>& p) { inv.insert(std::make_pair(p.second, p.first)); });
  return inv;
}

std::unordered_map<GradientEditMode, Serialize::GradientEditMode_PID> g_GradientModeToPermId = {
  { GradientEditMode::WINDOW_LEVEL, Serialize::GradientEditMode_PID::WINDOW_LEVEL },
  { GradientEditMode::ISOVALUE, Serialize::GradientEditMode_PID::ISOVALUE },
  { GradientEditMode::PERCENTILE, Serialize::GradientEditMode_PID::PERCENTILE },
  { GradientEditMode::CUSTOM, Serialize::GradientEditMode_PID::CUSTOM },
  { GradientEditMode::MINMAX, Serialize::GradientEditMode_PID::MINMAX },
};
auto g_PermIdToGradientMode = inverse_map(g_GradientModeToPermId);

std::unordered_map<eRenderDurationType, Serialize::DurationType_PID> g_RenderDurationTypeToPermId = {
  { eRenderDurationType::SAMPLES, Serialize::DurationType_PID::SAMPLES },
  { eRenderDurationType::TIME, Serialize::DurationType_PID::TIME },
};
auto g_PermIdToRenderDurationType = inverse_map(g_RenderDurationTypeToPermId);

std::unordered_map<ProjectionMode, Serialize::Projection_PID> g_ProjectionToPermId = {
  { ProjectionMode::PERSPECTIVE, Serialize::Projection_PID::PERSPECTIVE },
  { ProjectionMode::ORTHOGRAPHIC, Serialize::Projection_PID::ORTHOGRAPHIC },
};
auto g_PermIdToProjection = inverse_map(g_ProjectionToPermId);

std::unordered_map<int, Serialize::LightType> g_LightTypeToPermId = {
  { 1, Serialize::LightType::SKY },
  { 0, Serialize::LightType::AREA },
};
auto g_PermIdToLightType = inverse_map(g_LightTypeToPermId);

Serialize::ViewerState
stateFromJson(const nlohmann::json& jsonDoc)
{
  // VERSION MUST EXIST.  THROW OR PANIC IF NOT.
  std::array<uint32_t, 3> v = { 0, 0, 0 };
  if (jsonDoc.contains("version")) {
    auto ja = jsonDoc["version"];
    v[0] = ja.at(0).get<uint32_t>();
    v[1] = ja.at(1).get<uint32_t>();
    v[2] = ja.at(2).get<uint32_t>();
  } else {
    // ERROR
  }

  Version version(v);

  // we will fill this in from the jsonDoc.
  Serialize::ViewerState stateV2;

  // version checks.  Parse old data structures here.
  if (version <= Version(1, 4, 1)) {
    Serialize::ViewerState_V1 stateV1 = jsonDoc.get<Serialize::ViewerState_V1>();
    // fill in this from the old data structure.
    stateV2 = fromV1(stateV1);
  } else {
    stateV2 = jsonDoc.get<Serialize::ViewerState>();
  }
  return stateV2;
}

QString
stateToPythonScript(const Serialize::ViewerState& s)
{
  std::string outFileName = LoadSpec::getFilename(s.datasets[0].url);

  std::ostringstream ss;
  ss << "# pip install agave_pyclient" << std::endl;
  ss << "# agave --server &" << std::endl;
  ss << "# python myscript.py" << std::endl << std::endl;
  ss << "import agave_pyclient as agave" << std::endl << std::endl;
  renderlib::RendererType rendererType = renderlib::RendererType_Pathtrace;
  if (s.rendererType == Serialize::RendererType_PID::RAYMARCH) {
    rendererType = renderlib::RendererType_Raymarch;
  }
  std::string mode = renderlib::rendererTypeToString(rendererType);
  ss << "r = agave.AgaveRenderer(mode=\"" << mode << "\")" << std::endl;

  std::string obj = "r.";

  // use whole loadspec to get multiresolution level, selected channels and sub-ROI
  // TODO reconcile subpath with index of multiresolution level.
  LoadDataCommandD loaddata;
  loaddata.m_path = s.datasets[0].url;
  loaddata.m_scene = s.datasets[0].scene;
  try {
    // TODO this is not necessarily a valid conversion!
    // dataset spec should probably store both level index and string subpath?
    // or only convert level index to string for zarr and store integer everywhere?
    loaddata.m_level = std::stoi(s.datasets[0].subpath);
  } catch (...) {
    // anything bad that happened in stoi should be ok to catch here
    // unless the subpath string is something REALLY crazy.
    loaddata.m_level = 0;
  }
  loaddata.m_time = s.datasets[0].time;
  loaddata.m_channels = std::vector<int32_t>(s.datasets[0].channels.begin(), s.datasets[0].channels.end());
  loaddata.m_xmin = s.datasets[0].clipRegion[0][0];
  loaddata.m_xmax = s.datasets[0].clipRegion[0][1];
  loaddata.m_ymin = s.datasets[0].clipRegion[1][0];
  loaddata.m_ymax = s.datasets[0].clipRegion[1][1];
  loaddata.m_zmin = s.datasets[0].clipRegion[2][0];
  loaddata.m_zmax = s.datasets[0].clipRegion[2][1];
  ss << obj << LoadDataCommand(loaddata).toPythonString() << std::endl;
  // TODO use window size or render window capture dims?
  ss << obj << SetResolutionCommand({ s.capture.width, s.capture.height }).toPythonString() << std::endl;
  ss << obj
     << SetBackgroundColorCommand({ s.backgroundColor[0], s.backgroundColor[1], s.backgroundColor[2] }).toPythonString()
     << std::endl;
  ss << obj << ShowBoundingBoxCommand({ s.showBoundingBox }).toPythonString() << std::endl;
  ss << obj << ShowScaleBarCommand({ s.showScaleBar }).toPythonString() << std::endl;
  ss << obj
     << SetBoundingBoxColorCommand({ s.boundingBoxColor[0], s.boundingBoxColor[1], s.boundingBoxColor[2] })
          .toPythonString()
     << std::endl;
  // TODO use value from viewport or render window capture settings?
  ss << obj << SetRenderIterationsCommand({ s.capture.samples }).toPythonString() << std::endl;
  ss << obj << SetPrimaryRayStepSizeCommand({ s.pathTracer.primaryStepSize }).toPythonString() << std::endl;
  ss << obj << SetSecondaryRayStepSizeCommand({ s.pathTracer.secondaryStepSize }).toPythonString() << std::endl;
  ss << obj << SetInterpolationCommand({ s.interpolate }).toPythonString() << std::endl;
  ss << obj << SetVoxelScaleCommand({ s.scale[0], s.scale[1], s.scale[2] }).toPythonString() << std::endl;
  ss << obj
     << SetFlipAxisCommand({ s.flipAxis[0] < 0 ? -1 : 1, s.flipAxis[1] < 0 ? -1 : 1, s.flipAxis[2] < 0 ? -1 : 1 })
          .toPythonString()
     << std::endl;
  ss << obj
     << SetClipRegionCommand({ s.clipRegion[0][0],
                               s.clipRegion[0][1],
                               s.clipRegion[1][0],
                               s.clipRegion[1][1],
                               s.clipRegion[2][0],
                               s.clipRegion[2][1] })
          .toPythonString()
     << std::endl;
  if (s.clipPlane.enabled) {
    Plane p;
    p.normal.x = s.clipPlane.clipPlane[0];
    p.normal.y = s.clipPlane.clipPlane[1];
    p.normal.z = s.clipPlane.clipPlane[2];
    p.d = s.clipPlane.clipPlane[3];
    Transform3d tr;
    tr.m_center = { s.clipPlane.transform.translation[0],
                    s.clipPlane.transform.translation[1],
                    s.clipPlane.transform.translation[2] };
    // quat ctor is w,x,y,z
    tr.m_rotation = { s.clipPlane.transform.rotation[3],
                      s.clipPlane.transform.rotation[0],
                      s.clipPlane.transform.rotation[1],
                      s.clipPlane.transform.rotation[2] };
    p = p.transform(tr);
    ss << obj << SetClipPlaneCommand({ p.normal.x, p.normal.y, p.normal.z, p.d }).toPythonString() << std::endl;
  } else {
    ss << obj << SetClipPlaneCommand({ 0, 0, 0, 0 }).toPythonString() << std::endl;
  }
  ss << obj << SetCameraPosCommand({ s.camera.eye[0], s.camera.eye[1], s.camera.eye[2] }).toPythonString() << std::endl;
  ss << obj << SetCameraTargetCommand({ s.camera.target[0], s.camera.target[1], s.camera.target[2] }).toPythonString()
     << std::endl;
  ss << obj << SetCameraUpCommand({ s.camera.up[0], s.camera.up[1], s.camera.up[2] }).toPythonString() << std::endl;
  ss << obj
     << SetCameraProjectionCommand(
          { g_PermIdToProjection[s.camera.projection],
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
      case GradientEditMode::MINMAX:
        ss << obj << SetMinMaxThresholdCommand({ i, ch.lutParams.minu16, ch.lutParams.maxu16 }).toPythonString()
           << std::endl;
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
    std::vector<float> v;
    for (auto p : ch.colorMap.stops) {
      v.push_back(p.x);
      v.push_back(p.value[0]);
      v.push_back(p.value[1]);
      v.push_back(p.value[2]);
      v.push_back(p.value[3]);
    }
    ss << obj << SetColorRampCommand({ i, ch.colorMap.name, v }).toPythonString() << std::endl;
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

  ss << obj << SessionCommand({ outFileName + ".png" }).toPythonString() << std::endl;
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
  spec.isImageSequence = s.isImageSequence;
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

ColorRamp
stateToColorRamp(const Serialize::ViewerState& state, int channelIndex)
{
  ColorRamp cr;
  const auto& ch = state.channels[channelIndex];
  const auto& cm = ch.colorMap;
  cr.m_name = cm.name;
  for (size_t i = 0; i < cm.stops.size(); i++) {
    ColorControlPoint cp(
      cm.stops[i].x, cm.stops[i].value[0], cm.stops[i].value[1], cm.stops[i].value[2], cm.stops[i].value[3]);
    cr.m_stops.push_back(cp);
  }
  return cr;
}

GradientData
stateToGradientData(const Serialize::ViewerState& state, int channelIndex)
{
  GradientData gd;
  const auto& ch = state.channels[channelIndex];
  const auto& lut = ch.lutParams;
  gd.m_activeMode = g_PermIdToGradientMode[lut.mode];
  gd.m_window = lut.window;
  gd.m_level = lut.level;
  gd.m_isovalue = lut.isovalue;
  gd.m_isorange = lut.isorange;
  gd.m_pctLow = lut.pctLow;
  gd.m_pctHigh = lut.pctHigh;
  for (size_t i = 0; i < lut.controlPoints.size(); i += 5) {
    LutControlPoint cp;
    cp.first = lut.controlPoints[i].x;
    // note: only the last value of the vector is used currently
    cp.second = lut.controlPoints[i].value[3];
    gd.m_customControlPoints.push_back(cp);
  }
  return gd;
}

Light
stateToLight(const Serialize::ViewerState& state, int lightIndex)
{
  Light lt;
  const Serialize::LightSettings_V1& l = state.lights[lightIndex];
  lt.m_T = g_PermIdToLightType[l.type];
  lt.m_Distance = l.distance;
  lt.m_Theta = l.theta;
  lt.m_Phi = l.phi;
  lt.m_ColorTop = glm::make_vec3(l.topColor.data());
  lt.m_ColorMiddle = glm::make_vec3(l.middleColor.data());
  lt.m_ColorBottom = glm::make_vec3(l.bottomColor.data());
  lt.m_Color = glm::make_vec3(l.color.data());
  lt.m_ColorTopIntensity = l.topColorIntensity;
  lt.m_ColorMiddleIntensity = l.middleColorIntensity;
  lt.m_ColorBottomIntensity = l.bottomColorIntensity;
  lt.m_ColorIntensity = l.colorIntensity;
  lt.m_Width = l.width;
  lt.m_Height = l.height;

  return lt;
}
Serialize::LoadSettings
fromLoadSpec(const LoadSpec& loadSpec)
{
  Serialize::LoadSettings s;
  s.url = loadSpec.filepath;
  s.subpath = loadSpec.subpath;
  s.isImageSequence = loadSpec.isImageSequence;
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
  s.type = g_LightTypeToPermId[lt.m_T];
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

Serialize::CaptureSettings
fromCaptureSettings(const CaptureSettings& cs, int viewWidth, int viewHeight)
{
  Serialize::CaptureSettings s;
  // make sure capture settings are initialized to window size until
  // render dialog has been opened.
  s.width = cs.width == 0 ? viewWidth : cs.width;
  s.height = cs.height == 0 ? viewHeight : cs.height;
  s.samples = cs.renderDuration.samples;
  s.seconds = cs.renderDuration.duration;
  s.durationType = g_RenderDurationTypeToPermId[cs.renderDuration.durationType];
  s.startTime = cs.startTime;
  s.endTime = cs.endTime;
  s.outputDirectory = cs.outputDir;
  s.filenamePrefix = cs.filenamePrefix;
  return s;
}

Serialize::ColorMap
fromColorRamp(const ColorRamp& cr)
{
  Serialize::ColorMap s;
  s.name = cr.m_name;
  for (const auto& cp : cr.m_stops) {
    Serialize::ControlPointSettings_V1 c;
    c.x = cp.first;
    c.value = { (float)cp.r / 255.0f, (float)cp.g / 255.0f, (float)cp.b / 255.0f, (float)cp.a / 255.0f };
    s.stops.push_back(c);
  }
  return s;
}

Serialize::LutParams_V1
fromGradientData(const GradientData& gd)
{
  Serialize::LutParams_V1 s;
  s.mode = g_GradientModeToPermId[gd.m_activeMode];
  s.window = gd.m_window;
  s.level = gd.m_level;
  s.isovalue = gd.m_isovalue;
  s.isorange = gd.m_isorange;
  s.pctLow = gd.m_pctLow;
  s.pctHigh = gd.m_pctHigh;
  for (const auto& cp : gd.m_customControlPoints) {
    Serialize::ControlPointSettings_V1 c;
    c.x = cp.first;
    c.value = { cp.second, cp.second, cp.second, cp.second };
    s.controlPoints.push_back(c);
  }
  return s;
}