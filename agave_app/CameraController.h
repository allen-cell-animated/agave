#pragma once

#include "Camera.h"
#include "Flags.h"

#include "renderlib/CCamera.h"
#include "renderlib/gesture/gesture.h"

#include <glm/glm.hpp>

class RenderSettings;
class QMouseEvent;

struct CameraModifier
{
  glm::vec3 position = { 0, 0, 0 };
  glm::vec3 target = { 0, 0, 0 };
  glm::vec3 up = { 0, 0, 0 };
  // float fov = 0;
  float nearClip = 0, farClip = 0;
};

inline CameraModifier
operator+(const CameraModifier& a, const CameraModifier& b)
{
  CameraModifier c;
  c.position = a.position + b.position;
  c.target = a.target + b.target;
  c.up = a.up + b.up;
  // c.fov = a.fov + b.fov;
  c.nearClip = a.nearClip + b.nearClip;
  c.farClip = a.farClip + b.farClip;
  return c;
}

inline CameraModifier
operator*(const CameraModifier& a, const float b)
{
  CameraModifier c;
  c.position = a.position * b;
  c.target = a.target * b;
  c.up = a.up * b;
  // c.fov = a.fov * b;
  c.nearClip = a.nearClip * b;
  c.farClip = a.farClip * b;
  return c;
}

struct CameraAnimation
{
  float duration; //< animation total time
  float time;     //< animation current time
  CameraModifier mod;
};

// Define interaction style for controlling a realistic camera
class CameraController
{
public:
  CameraController(QCamera* cam, CCamera* theCam);

  enum EMouseButtonFlag
  {
    Left = 0x0001,
    Middle = 0x0002,
    Right = 0x0004
  };

  void setRenderSettings(RenderSettings& rs) { m_renderSettings = &rs; }

  virtual void OnMouseWheelForward(void);
  virtual void OnMouseWheelBackward(void);
  virtual void OnMouseMove(QMouseEvent* event);

  int m_OldPos[2];
  int m_NewPos[2];

  // Camera sensitivity to mouse movement
  static float m_OrbitSpeed;
  static float m_PanSpeed;
  static float m_ZoomSpeed;
  static float m_ContinuousZoomSpeed;
  static float m_ApertureSpeed;
  static float m_FovSpeed;

  RenderSettings* m_renderSettings;
  QCamera* m_qcamera;
  CCamera* m_CCamera;
};

extern bool
cameraManipulation(const glm::vec2 viewportSize,
                   // const TimeSample& clock,
                   Gesture& gesture,
                   CCamera& camera,
                   CameraModifier& cameraMod);

inline CCamera&
operator+(CCamera& camera, const CameraModifier& mod)
{
  camera.m_From += mod.position;
  camera.m_Target += mod.target;
  camera.m_Up += mod.up;
  // camera.m_FovV += mod.fov;
  camera.m_Near += mod.nearClip;
  camera.m_Far += mod.farClip;
  // camera.Update();
  return camera;
}