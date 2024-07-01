#ifndef OFFSCREEN_RENDERER_H
#define OFFSCREEN_RENDERER_H
#pragma once

#include "glad/glad.h"

#include "RenderInterface.h"
#include "command.h"
#include "renderlib/gesture/gesture.h"
#include "renderlib/graphics/IRenderWindow.h"
#include "renderlib/graphics/gl/Util.h"
#include "renderlib/renderlib.h"

#include <QList>
#include <QObject>

#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QOpenGLFramebufferObject>
#include <QOpenGLTexture>
#include <QThread>

#include <memory>
#include <string>

class commandBuffer;
class CCamera;
class ImageXYZC;
class RenderSettings;
class Scene;

class OffscreenRenderer : public RenderInterface
{
public:
  OffscreenRenderer();
  virtual ~OffscreenRenderer();

  // RenderInterface

  // tell server to identify this session?
  virtual int Session(const std::string&);
  // tell server where files might be (appends to existing)
  virtual int AssetPath(const std::string&);
  // load a volume
  virtual int LoadOmeTif(const std::string&);
  // load a volume
  virtual int LoadVolumeFromFile(const std::string&, int, int);
  // change load same volume file, different time index
  virtual int SetTime(int);
  // set camera pos
  virtual int Eye(float, float, float);
  // set camera target pt
  virtual int Target(float, float, float);
  // set camera up direction
  virtual int Up(float, float, float);
  virtual int Aperture(float);
  // perspective(0)/ortho(1), fov(degrees)/orthoscale(world units)
  virtual int CameraProjection(int32_t, float);
  virtual int Focaldist(float);
  virtual int Exposure(float);
  virtual int MatDiffuse(int32_t, float, float, float, float);
  virtual int MatSpecular(int32_t, float, float, float, float);
  virtual int MatEmissive(int32_t, float, float, float, float);
  // set num render iterations
  virtual int RenderIterations(int32_t);
  // (continuous or on-demand frames)
  virtual int StreamMode(int32_t);
  // request new image
  virtual int Redraw();
  virtual int SetResolution(int32_t, int32_t);
  virtual int Density(float);
  // move camera to bound and look at the scene contents
  virtual int FrameScene();
  virtual int MatGlossiness(int32_t, float);
  // channel index, 1/0 for enable/disable
  virtual int EnableChannel(int32_t, int32_t);
  // channel index, window, level.  (Do I ever set these independently?)
  virtual int SetWindowLevel(int32_t, float, float);
  // theta, phi in degrees
  virtual int OrbitCamera(float, float);
  virtual int TrackballCamera(float, float);
  virtual int SkylightTopColor(float, float, float);
  virtual int SkylightMiddleColor(float, float, float);
  virtual int SkylightBottomColor(float, float, float);
  // r, theta, phi
  virtual int LightPos(int32_t, float, float, float);
  virtual int LightColor(int32_t, float, float, float);
  // x by y size
  virtual int LightSize(int32_t, float, float);
  // xmin, xmax, ymin, ymax, zmin, zmax
  virtual int SetClipRegion(float, float, float, float, float, float);
  // x, y, z pixel scaling
  virtual int SetVoxelScale(float, float, float);
  // channel, method
  virtual int AutoThreshold(int32_t, int32_t);
  // channel index, pct_low, pct_high.  (Do I ever set these independently?)
  virtual int SetPercentileThreshold(int32_t, float, float);
  virtual int MatOpacity(int32_t, float);
  virtual int SetPrimaryRayStepSize(float);
  virtual int SetSecondaryRayStepSize(float);
  virtual int BackgroundColor(float, float, float);
  virtual int SetIsovalueThreshold(int32_t, float, float);
  virtual int SetControlPoints(int32_t, std::vector<float>);
  virtual int SetBoundingBoxColor(float, float, float);
  virtual int ShowBoundingBox(int32_t);
  virtual int ShowScaleBar(int32_t);
  virtual int SetFlipAxis(int32_t, int32_t, int32_t);
  virtual int SetInterpolation(int32_t);

protected:
  void init();

  void resizeGL(int internalWidth, int internalHeight);

  QImage render();

  void reset(int from = 0);

  void shutDown();

private:
#if HAS_EGL
  HeadlessGLContext* m_glContext;
#else
  QOpenGLContext* m_glContext;
  QOffscreenSurface* m_surface;
#endif

  GLFramebufferObject* m_fbo;

  int32_t m_width, m_height;

  // TODO move this info.  This class only knows about some abstract renderer and a scene object.
  void myVolumeInit();
  struct myVolumeData
  {
    RenderSettings* m_renderSettings;
    IRenderWindow* m_renderer;
    Scene* m_scene;
    CCamera* m_camera;
    Gesture m_gesture;

    myVolumeData()
      : m_camera(nullptr)
      , m_scene(nullptr)
      , m_renderSettings(nullptr)
      , m_renderer(nullptr)
    {
    }
  } m_myVolumeData;

  ExecutionContext m_ec;

  QImage m_lastRenderedImage;
  std::string m_session;
};

#endif // OFFSCREEN_RENDERER_H
