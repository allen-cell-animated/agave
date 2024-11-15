#pragma once

#include "BoundingBox.h"
#include "Defines.h"
#include "MathUtil.h"
#include "glm.h"

#define DEF_FOCUS_TYPE CenterScreen
#define DEF_FOCUS_SENSOR_POS_CANVAS glm::vec2(0.0f)
#define DEF_FOCUS_P glm::vec3(0.0f)
#define DEF_FOCUS_FOCAL_DISTANCE 100.0f
#define DEF_FOCUS_T 0.0f
#define DEF_FOCUS_N glm::vec3(0.0f)
#define DEF_FOCUS_DOT_WN 0.0f

enum ProjectionMode
{
  PERSPECTIVE,
  ORTHOGRAPHIC
};

class Focus
{
public:
  enum EType
  {
    CenterScreen,
    ScreenPoint,
    Probed,
    Manual
  };

  EType m_Type;
  glm::vec2 m_SensorPosCanvas;
  float m_FocalDistance;
  float m_T;
  glm::vec3 m_P;
  glm::vec3 m_N;
  float m_DotWN;

  Focus(void)
  {
    m_Type = DEF_FOCUS_TYPE;
    m_SensorPosCanvas = DEF_FOCUS_SENSOR_POS_CANVAS;
    m_FocalDistance = DEF_FOCUS_FOCAL_DISTANCE;
    m_T = DEF_FOCUS_T;
    m_P = DEF_FOCUS_P;
    m_N = DEF_FOCUS_N;
    m_DotWN = DEF_FOCUS_DOT_WN;
  }

  Focus& operator=(const Focus& Other)
  {
    m_Type = Other.m_Type;
    m_SensorPosCanvas = Other.m_SensorPosCanvas;
    m_FocalDistance = Other.m_FocalDistance;
    m_P = Other.m_P;
    m_T = Other.m_T;
    m_N = Other.m_N;
    m_DotWN = Other.m_DotWN;

    return *this;
  }
};

#define DEF_APERTURE_SIZE 0.0f
#define DEF_APERTURE_NO_BLADES 5
#define DEF_APERTURE_BIAS BiasNone
#define DEF_APERTURE_ROTATION 0.0f

#define MAX_BOKEH_DATA (12)

class Aperture
{
public:
  enum EBias
  {
    BiasCenter,
    BiasEdge,
    BiasNone
  };

  float m_Size;
  int m_NoBlades;
  EBias m_Bias;
  float m_Rotation;
  float m_Data[MAX_BOKEH_DATA];

  Aperture(void)
  {
    m_Size = DEF_APERTURE_SIZE;
    m_NoBlades = DEF_APERTURE_NO_BLADES;
    m_Bias = DEF_APERTURE_BIAS;
    m_Rotation = DEF_APERTURE_ROTATION;

    for (int i = 0; i < MAX_BOKEH_DATA; i++)
      m_Data[i] = 0.0f;
  }

  Aperture& operator=(const Aperture& Other)
  {
    m_Size = Other.m_Size;
    m_NoBlades = Other.m_NoBlades;
    m_Bias = Other.m_Bias;
    m_Rotation = Other.m_Rotation;

    for (int i = 0; i < MAX_BOKEH_DATA; i++)
      m_Data[i] = Other.m_Data[i];

    return *this;
  }

  void Update(const float& FStop)
  {
    // Update bokeh
    int Ns = (int)m_NoBlades;

    if ((Ns >= 3) && (Ns <= 6)) {
      float w = m_Rotation * PI_F / 180.0f, wi = (2.0f * PI_F) / (float)Ns;

      Ns = (Ns + 2) * 2;

      for (int i = 0; i < Ns && i < MAX_BOKEH_DATA; i += 2) {
        m_Data[i] = cos(w);
        m_Data[i + 1] = sin(w);
        w += wi;
      }
    }
  }
};

class Resolution2D
{
public:
  Resolution2D(const float& Width, const float& Height)
  {
    m_XY = glm::ivec2(Width, Height);

    Update();
  }

  Resolution2D(void)
  {
    m_XY = glm::ivec2(640, 480);

    Update();
  }

  ~Resolution2D(void) {}

  Resolution2D& operator=(const Resolution2D& Other)
  {
    m_XY = Other.m_XY;
    m_InvXY = Other.m_InvXY;
    m_NoElements = Other.m_NoElements;
    m_AspectRatio = Other.m_AspectRatio;
    m_DiagonalLength = Other.m_DiagonalLength;

    return *this;
  }

  int operator[](int i) const { return m_XY[i]; }

  int& operator[](int i) { return m_XY[i]; }

  bool operator==(const Resolution2D& Other) const
  {
    return GetResX() == Other.GetResX() && GetResY() == Other.GetResY();
  }

  bool operator!=(const Resolution2D& Other) const
  {
    return GetResX() != Other.GetResX() || GetResY() != Other.GetResY();
  }

  void Update(void)
  {
    m_InvXY = glm::vec2(1.0f / m_XY.x, 1.0f / m_XY.y);
    m_NoElements = m_XY.x * m_XY.y;
    m_AspectRatio = (float)m_XY.x / (float)m_XY.y;
    m_DiagonalLength = sqrtf(powf((float)m_XY.x, 2.0f) + powf((float)m_XY.y, 2.0f));
  }

  glm::ivec2 ToVector(void) const { return glm::ivec2(m_XY.x, m_XY.y); }

  void Set(const glm::ivec2& Resolution)
  {
    m_XY = Resolution;

    Update();
  }

  int GetResX(void) const { return m_XY.x; }
  void SetResX(const int& Width)
  {
    m_XY.x = Width;
    Update();
  }
  int GetResY(void) const { return m_XY.y; }
  void SetResY(const int& Height)
  {
    m_XY.y = Height;
    Update();
  }
  glm::vec2 GetInv(void) const { return m_InvXY; }
  int GetNoElements(void) const { return m_NoElements; }
  float GetAspectRatio(void) const { return m_AspectRatio; }

  void PrintSelf(void) const { printf("[%d x %d]\n", GetResX(), GetResY()); }

private:
  glm::ivec2 m_XY;        /*!< Resolution width and height */
  glm::vec2 m_InvXY;      /*!< Resolution width and height reciprocal */
  int m_NoElements;       /*!< No. elements */
  float m_AspectRatio;    /*!< Aspect ratio of image plane */
  float m_DiagonalLength; /*!< Diagonal length */
};

#define DEF_FILM_ISO 400.0f
#define DEF_FILM_EXPOSURE 0.25f
#define DEF_FILM_FSTOP 8.0f
#define DEF_FILM_GAMMA 2.2f

class Film
{
public:
  Resolution2D m_Resolution;
  float m_Screen[2][2]; // [left, right], [bottom, top]
  glm::vec2 m_InvScreen;
  float m_Iso;
  float m_Exposure;
  int m_ExposureIterations;
  float m_FStop;
  float m_Gamma;

  // ToDo: Add description
  Film(void)
  {
    m_Screen[0][0] = 0.0f;
    m_Screen[0][1] = 0.0f;
    m_Screen[1][0] = 0.0f;
    m_Screen[1][1] = 0.0f;
    m_InvScreen = glm::vec2(0.0f);
    m_Iso = DEF_FILM_ISO;
    m_Exposure = DEF_FILM_EXPOSURE;
    m_ExposureIterations = 1;
    m_FStop = DEF_FILM_FSTOP;
    m_Gamma = DEF_FILM_GAMMA;
  }

  Film& operator=(const Film& Other)
  {
    m_Resolution = Other.m_Resolution;
    m_Screen[0][0] = Other.m_Screen[0][0];
    m_Screen[0][1] = Other.m_Screen[0][1];
    m_Screen[1][0] = Other.m_Screen[1][0];
    m_Screen[1][1] = Other.m_Screen[1][1];
    m_InvScreen = Other.m_InvScreen;
    m_Iso = Other.m_Iso;
    m_Exposure = Other.m_Exposure;
    m_ExposureIterations = Other.m_ExposureIterations;
    m_FStop = Other.m_FStop;
    m_Gamma = Other.m_Gamma;

    return *this;
  }

  void Update(const float& FovV, const float& Aperture, const ProjectionMode& projection, const float& orthoScale)
  {
    float Scale = 0.0f;

    Scale = (projection == ORTHOGRAPHIC) ? orthoScale : tanf(0.5f * (FovV * DEG_TO_RAD));

    // left, right
    m_Screen[0][0] = -Scale * m_Resolution.GetAspectRatio();
    m_Screen[0][1] = Scale * m_Resolution.GetAspectRatio();
    // the "0" Y pixel will be at -Scale. this is the BOTTOM of the screen
    // bottom, top
    m_Screen[1][0] = -Scale;
    m_Screen[1][1] = Scale;

    // the amount to increment for each pixel
    m_InvScreen.x = (m_Screen[0][1] - m_Screen[0][0]) / m_Resolution.GetResX();
    m_InvScreen.y = (m_Screen[1][1] - m_Screen[1][0]) / m_Resolution.GetResY();

    m_Resolution.Update();
  }

  int GetWidth(void) const { return m_Resolution.GetResX(); }

  int GetHeight(void) const { return m_Resolution.GetResY(); }
};

#define FPS1 30.0f

// #define DEF_CAMERA_TYPE						Perspective
#define DEF_CAMERA_OPERATOR CameraOperatorUndefined
#define DEF_CAMERA_VIEW_MODE ViewModeBack
#define DEF_CAMERA_NEAR 0.01f
#define DEF_CAMERA_FAR 20.0f
#define DEF_CAMERA_ENABLE_CLIPPING true
#define DEF_CAMERA_GAMMA 2.2f
#define DEF_CAMERA_FIELD_OF_VIEW 55.0f
#define DEF_CAMERA_NUM_APERTURE_BLADES 4
#define DEF_CAMERA_APERTURE_BLADES_ANGLE 0.0f
#define DEF_CAMERA_ASPECT_RATIO 1.0f
#define DEF_ORTHO_SCALE 0.5f
// #define DEF_CAMERA_ZOOM_SPEED				1.0f
// #define DEF_CAMERA_ORBIT_SPEED				5.0f
// #define DEF_CAMERA_APERTURE_SPEED			0.25f
// #define DEF_CAMERA_FOCAL_DISTANCE_SPEED		10.0f

class CCamera
{
public:
  CBoundingBox m_SceneBoundingBox;
  float m_Near;
  float m_Far;
  bool m_EnableClippingPlanes;
  glm::vec3 m_From;
  glm::vec3 m_Target;
  glm::vec3 m_Up;
  float m_FovV;
  float m_AreaPixel;

  // vector corresponding to into the screen (means N, U, V is left handed!)
  glm::vec3 m_N;
  // vector corresponding to screen left to right
  glm::vec3 m_U;
  // vector corresponding to screen bottom to top
  glm::vec3 m_V;

  Film m_Film;
  Focus m_Focus;
  Aperture m_Aperture;
  bool m_Dirty;
  ProjectionMode m_Projection;
  float m_OrthoScale;

  CCamera(void)
  {
    m_Near = DEF_CAMERA_NEAR;
    m_Far = DEF_CAMERA_FAR;
    m_EnableClippingPlanes = DEF_CAMERA_ENABLE_CLIPPING;
    m_From = glm::vec3(500.0f, 500.0f, 500.0f);
    m_Target = glm::vec3(0.0f, 0.0f, 0.0f);
    m_Up = glm::vec3(0.0f, 1.0f, 0.0f);
    m_FovV = DEF_CAMERA_FIELD_OF_VIEW;
    m_N = glm::vec3(0.0f, 0.0f, 1.0f);
    m_U = glm::vec3(1.0f, 0.0f, 0.0f);
    m_V = glm::vec3(0.0f, 1.0f, 0.0f);
    m_Dirty = true;
    m_Projection = PERSPECTIVE;
    m_OrthoScale = DEF_ORTHO_SCALE;
    m_AreaPixel = m_Film.m_Resolution.GetAspectRatio() / (m_Focus.m_FocalDistance * m_Focus.m_FocalDistance);
  }
  CCamera(const CCamera& other)
  {
    m_SceneBoundingBox = other.m_SceneBoundingBox;
    m_Near = other.m_Near;
    m_Far = other.m_Far;
    m_EnableClippingPlanes = other.m_EnableClippingPlanes;
    m_From = other.m_From;
    m_Target = other.m_Target;
    m_Up = other.m_Up;
    m_FovV = other.m_FovV;
    m_AreaPixel = other.m_AreaPixel;
    m_N = other.m_N;
    m_U = other.m_U;
    m_V = other.m_V;
    m_Film = other.m_Film;
    m_Focus = other.m_Focus;
    m_Aperture = other.m_Aperture;
    m_Dirty = other.m_Dirty;
    m_Projection = other.m_Projection;
    m_OrthoScale = other.m_OrthoScale;
  }
  CCamera& operator=(const CCamera& Other)
  {
    m_SceneBoundingBox = Other.m_SceneBoundingBox;
    m_Near = Other.m_Near;
    m_Far = Other.m_Far;
    m_EnableClippingPlanes = Other.m_EnableClippingPlanes;
    m_From = Other.m_From;
    m_Target = Other.m_Target;
    m_Up = Other.m_Up;
    m_FovV = Other.m_FovV;
    m_AreaPixel = Other.m_AreaPixel;
    m_N = Other.m_N;
    m_U = Other.m_U;
    m_V = Other.m_V;
    m_Film = Other.m_Film;
    m_Focus = Other.m_Focus;
    m_Aperture = Other.m_Aperture;
    m_Dirty = Other.m_Dirty;
    m_Projection = Other.m_Projection;
    m_OrthoScale = Other.m_OrthoScale;

    return *this;
  }

  void Update(void)
  {
    // right handed coordinate system

    // "z" lookat direction
    m_N = glm::normalize(m_Target - m_From);
    // camera left/right
    m_U = glm::normalize(glm::cross(m_N, m_Up));
    // camera up/down
    m_V = glm::normalize(glm::cross(m_U, m_N));

    m_Film.Update(m_FovV, m_Aperture.m_Size, m_Projection, m_OrthoScale);

    m_AreaPixel = m_Film.m_Resolution.GetAspectRatio() / (m_Focus.m_FocalDistance * m_Focus.m_FocalDistance);

    m_Aperture.Update(m_Film.m_FStop);

    m_Film.Update(m_FovV, m_Aperture.m_Size, m_Projection, m_OrthoScale);
  }

  // use with fixed amount for scroll wheel zooming
  void Zoom(float amount)
  {
    glm::vec3 reverseLoS = m_From - m_Target;

    if (amount > 0) {
      float factor = 1.1f;
      reverseLoS *= factor;
      m_OrthoScale *= factor;
    } else if (amount < 0) {
      if (glm::length(reverseLoS) > 0.0005f) {
        float factor = 0.9f;
        reverseLoS *= factor;
        m_OrthoScale *= factor;
      }
    }

    m_From = reverseLoS + m_Target;
  }

  // Pan operator
  void Pan(float UpUnits, float RightUnits)
  {
    glm::vec3 LoS = m_Target - m_From;

    glm::vec3 right = glm::cross(LoS, m_Up);
    glm::vec3 orthogUp = glm::cross(right, LoS);

    right = glm::normalize(right);
    orthogUp = glm::normalize(orthogUp);

    const float Length = glm::length(m_Target - m_From);

    const unsigned int WindowWidth = m_Film.m_Resolution.GetResX();

    const float U = Length * (RightUnits / WindowWidth);
    const float V = Length * (UpUnits / WindowWidth);

    m_From = m_From + right * U + m_Up * V;
    m_Target = m_Target + right * U + m_Up * V;
  }

  void Orbit(float DownDegrees, float RightDegrees)
  {
    glm::vec3 ReverseLoS = m_From - m_Target;

    glm::vec3 right = glm::cross(m_Up, ReverseLoS);
    // glm::vec3 orthogUp = glm::cross(ReverseLoS, right);
    glm::vec3 Up = glm::vec3(0.0f, 1.0f, 0.0f);

    ReverseLoS = glm::rotate(ReverseLoS, DownDegrees * DEG_TO_RAD, right);
    ReverseLoS = glm::rotate(ReverseLoS, RightDegrees * DEG_TO_RAD, Up);
    m_Up = glm::rotate(m_Up, DownDegrees * DEG_TO_RAD, right);
    m_Up = glm::rotate(m_Up, RightDegrees * DEG_TO_RAD, Up);

    m_From = ReverseLoS + m_Target;
  }

  void Trackball(float DownDegrees, float RightDegrees)
  {
    float angle = sqrtf(DownDegrees * DownDegrees + RightDegrees * RightDegrees);
    angle = DEG_TO_RAD * angle;

    glm::vec3 _eye = m_From - m_Target;

    glm::vec3 objectUpDirection = m_Up; // or m_V; ???
    glm::vec3 objectSidewaysDirection = m_U;

    objectUpDirection *= DownDegrees;
    objectSidewaysDirection *= -RightDegrees;

    glm::vec3 moveDirection = objectUpDirection + objectSidewaysDirection;

    glm::vec3 axis = glm::normalize(glm::cross(moveDirection, _eye));

    _eye = glm::rotate(_eye, angle, axis);
    m_Up = glm::rotate(m_Up, angle, axis);

    m_From = _eye + m_Target;
  }

  void SetProjectionMode(const ProjectionMode projectionMode)
  {
    m_Projection = projectionMode;
    Update();
  }

  void SetViewMode(const EViewMode ViewMode);

  float getHalfHorizontalAperture() const { return tan(this->GetHorizontalFOV_radians() * 0.5f); }

  float GetHorizontalFOV_radians() const
  {
    // convert horz fov to vert fov
    // w/d = 2*tan(hfov/2)
    // h/d = 2*tan(vfov/2)
    float hfov = 2.0f * atan((float)m_Film.GetWidth() / (float)m_Film.GetHeight() * tan(m_FovV * 0.5f * DEG_TO_RAD));
    return hfov;
  }
  float GetVerticalFOV_radians() const { return m_FovV * DEG_TO_RAD; }

  void getViewMatrix(glm::mat4& viewMatrix) const
  {
    // TODO future just do this inside of Update()

    glm::vec3 eye(m_From.x, m_From.y, m_From.z);
    glm::vec3 center(m_Target.x, m_Target.y, m_Target.z);
    glm::vec3 up(m_Up.x, m_Up.y, m_Up.z);
    viewMatrix = glm::lookAt(eye, center, up);
  }

  // return the world-space vectors that correspond to camera x, y, z directions
  LinearSpace3f getFrame() const;

  float getDistance(glm::vec3 p) const { return glm::distance(p, m_From); }

  void getProjMatrix(glm::mat4& projMatrix) const
  {
    // TODO future just do this inside of Update()

    float w = (float)m_Film.GetWidth();
    float h = (float)m_Film.GetHeight();
    float vfov = m_FovV * DEG_TO_RAD;

    if (m_Projection == PERSPECTIVE) {
      projMatrix = glm::perspectiveFov(vfov, w, h, m_Near, m_Far);
    } else {
      projMatrix = glm::ortho(
        -(w / h) * m_OrthoScale, (w / h) * m_OrthoScale, -1.0f * m_OrthoScale, 1.0f * m_OrthoScale, m_Near, m_Far);
    }
  }

  void ComputeFitToBounds(const CBoundingBox& sceneBBox, glm::vec3& newPosition, glm::vec3& newTarget) const;
};

struct CameraModifier
{
  glm::vec3 position = { 0, 0, 0 };
  glm::vec3 target = { 0, 0, 0 };
  glm::vec3 up = { 0, 0, 0 };
  float fov = 0;
  float nearClip = 0, farClip = 0;

  CameraModifier()
    : nearClip(0)
    , farClip(0)
  {
  }
};

inline CameraModifier
operator+(const CameraModifier& a, const CameraModifier& b)
{
  CameraModifier c;
  c.position = a.position + b.position;
  c.target = a.target + b.target;
  c.up = a.up + b.up;
  c.fov = a.fov + b.fov;
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
  c.fov = a.fov * b;
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

struct Gesture;

extern bool
cameraManipulation(const glm::vec2 viewportSize, Gesture& gesture, CCamera& camera, CameraModifier& cameraMod);

inline CCamera&
operator+=(CCamera& camera, const CameraModifier& mod)
{
  // update OrthoScale as well - remember percentage change in distance
  // from target to eye is the same as percentage change in ortho scale
  float dold = glm::distance(camera.m_From, camera.m_Target);
  camera.m_From += mod.position;
  camera.m_Target += mod.target;
  float dnew = glm::distance(camera.m_From, camera.m_Target);
  float scale = dnew / dold;
  camera.m_OrthoScale *= scale;
  camera.m_Up += mod.up;
  // camera.m_FovV += mod.fov;
  camera.m_Near += mod.nearClip;
  camera.m_Far += mod.farClip;
  // camera.Update();
  return camera;
}

inline CCamera
operator+(const CCamera& camera, const CameraModifier& mod)
{
  // a new copy
  CCamera c = camera;
  // apply the mod to the new copy
  c += mod;
  return c;
}
