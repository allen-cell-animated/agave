#pragma once

#include "IRenderWindowBase.h"
#include <inttypes.h>

class CCamera;
class CStatus;
class GLFramebufferObject;
class RenderSettings;
class Scene;

#include <memory>

class IRenderWindow : public IRenderWindowBase
{
public:
  IRenderWindow();
  virtual ~IRenderWindow();

  // OpenGL-specific version of renderTo
  virtual void renderTo(const CCamera& camera, GLFramebufferObject* fbo) = 0;
};
