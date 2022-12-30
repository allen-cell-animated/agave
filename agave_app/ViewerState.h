#pragma once

#include "renderlib/GradientData.h" // for GradientEditMode enum

#include <glm.h>

#include <QJsonDocument>
#include <QJsonObject>
#include <QString>

#include <map>
#include <vector>

struct LutParams
{
  static std::map<GradientEditMode, int> g_GradientModeToPermId;
  static std::map<int, GradientEditMode> g_PermIdToGradientMode;

  float m_window = 1.0f, m_level = 0.5f;
  float m_isovalue = 0.5f, m_isorange = 0.1f;
  float m_pctLow = 0.5f, m_pctHigh = 0.98f;
  std::vector<LutControlPoint> m_customControlPoints;
  // permanent id value for serialization???
  int m_mode;
};

struct ChannelViewerState
{
  bool m_enabled = true;
  LutParams m_lutParams;
  float m_opacity = 1.0f;
  float m_glossiness = 0.0f;
  glm::vec3 m_diffuse = glm::vec3(0.5f, 0.5f, 0.5f), m_specular, m_emissive;
};

struct LightViewerState
{
  int m_type = 0;
  float m_theta = 0.0f, m_phi = 0.0f;
  float m_colorIntensity = 1.0;
  glm::vec3 m_color = glm::vec3(0.5f, 0.5f, 0.5f);
  glm::vec3 m_topColor = glm::vec3(0.5f, 0.5f, 0.5f), m_middleColor = glm::vec3(0.5f, 0.5f, 0.5f),
            m_bottomColor = glm::vec3(0.5, 0.5, 0.5);
  float m_topColorIntensity = 1.0;
  float m_middleColorIntensity = 1.0;
  float m_bottomColorIntensity = 1.0;
  float m_width = 1.0f, m_height = 1.0f, m_distance = 10.0f;
};

struct CaptureState
{
  std::string mOutputDir;
  std::string mFilenamePrefix;
  int mWidth;
  int mHeight;
  int mSamples;
  int mDuration; // in seconds
  // permanent id value for serialization???
  int mDurationType; // samples or seconds
  int mStartTime;
  int mEndTime;
};

struct ViewerState
{
  std::string m_volumeImageFile;
  std::vector<ChannelViewerState> m_channels;
  glm::vec3 m_backgroundColor;
  bool m_showBoundingBox;
  glm::vec3 m_boundingBoxColor;
  int m_resolutionX = 0, m_resolutionY = 0;
  int m_renderIterations = 1;
  float m_exposure = 0.75f;
  float m_densityScale = 50.0f;
  enum Projection
  {
    PERSPECTIVE,
    ORTHOGRAPHIC
  };
  int m_projection = Projection::PERSPECTIVE;
  float m_fov = 55.0f;
  float m_orthoScale = 1.0f;
  float m_apertureSize = 0.0f;
  float m_focalDistance = 0.0f;
  float m_gradientFactor = 0.5f;
  float m_primaryStepSize = 4.0f, m_secondaryStepSize = 4.0f;
  float m_roiXmax = 1.0f, m_roiYmax = 1.0f, m_roiZmax = 1.0f, m_roiXmin = 0.0f, m_roiYmin = 0.0f, m_roiZmin = 0.0f;
  float m_scaleX = 1.0f, m_scaleY = 1.0f, m_scaleZ = 1.0f;

  float m_eyeX, m_eyeY, m_eyeZ;
  float m_targetX, m_targetY, m_targetZ;
  float m_upX, m_upY, m_upZ;

  // timeline state
  int32_t m_minTime = 0, m_maxTime = 0, m_currentTime = 0, m_currentScene = 0;

  LightViewerState m_light0;
  LightViewerState m_light1;
  CaptureState m_captureState;

  QJsonDocument stateToJson() const;
  QString stateToPythonScript() const;

  void stateFromJson(QJsonDocument& jsonDoc);

private:
  LutParams lutParamsFromJson(QJsonObject& jsonObj);
};
