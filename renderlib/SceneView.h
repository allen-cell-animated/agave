#pragma once

#include "CCamera.h"

#include <memory>

class Scene;
class RenderSettings;
class Shaders;

// collect up a Scene, CCamera, and Viewport region for rendering
struct SceneView
{
  // viewport region is min value at lower left, max value at upper right of window
  struct Viewport
  {
    struct Region
    {
      Region()
        : lower(+INT_MAX)
        , upper(-INT_MAX)
      {
      }

      Region(const glm::ivec2& lower, const glm::ivec2& upper)
        : lower(lower)
        , upper(upper){};

      // assignment operator
      Region& operator=(const Region& other)
      {
        lower = other.lower;
        upper = other.upper;
        return *this;
      }

      void extend(const glm::ivec2& p)
      {
        lower = glm::min(lower, p);
        upper = glm::max(upper, p);
      }

      glm::ivec2 size() const { return upper - lower; }

      bool empty() const { return size().x < 0 || size().y < 0; }

      glm::ivec2 lower;
      glm::ivec2 upper;

      static Region intersect(const Region& a, const Region& b);
    };
    Region region;
    // flip Y to match window coordinates
    glm::ivec2 toRaster(const glm::vec2& p) const;
    // scale to the lower left (-1, -1) to upper right (1,1) range
    glm::vec2 toNDC(const glm::ivec2& p) const;
  } viewport;
  CCamera camera;
  std::unique_ptr<Shaders> shaders;
  Scene* scene = nullptr;
  RenderSettings* renderSettings = nullptr;

  bool anythingActive() const { return scene != nullptr; }
};
