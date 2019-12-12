#ifndef OFFSCREEN_RENDERER_H
#define OFFSCREEN_RENDERER_H
#pragma once

#include "glad/glad.h"

#include <QList>
#include <QObject>

#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QOpenGLFramebufferObject>
#include <QOpenGLTexture>
#include <QThread>

#include <memory>

class commandBuffer;
class CCamera;
class ImageXYZC;
class RenderGLPT;
class RenderSettings;
class Scene;

class OffscreenRenderer
{
public:
  OffscreenRenderer();
  virtual ~OffscreenRenderer();

  void init();

  void resizeGL(int internalWidth, int internalHeight);

protected:
  QImage render();

  void reset(int from = 0);

  void shutDown();

private:
  QOpenGLContext* context;
  QOffscreenSurface* surface;
  QOpenGLFramebufferObject* fbo;

  int32_t _width, _height;

  int frameNumber;

  // TODO move this info.  This class only knows about some abstract renderer and a scene object.
  void myVolumeInit();
  struct myVolumeData
  {
    RenderSettings* _renderSettings;
    RenderGLPT* _renderer;
    Scene* _scene;
    CCamera* _camera;

    myVolumeData()
      : _camera(nullptr)
      , _scene(nullptr)
      , _renderSettings(nullptr)
      , _renderer(nullptr)
    {}
  } myVolumeData;
};

#endif // OFFSCREEN_RENDERER_H
