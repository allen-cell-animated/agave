#pragma once
#include "IRenderWindow.h"

#include <memory>

class Image2D;
class ImageXYZC;

class RenderGL2d : public IRenderWindow
{
public:
  RenderGL2d(std::shared_ptr<ImageXYZC> img);
  virtual ~RenderGL2d();

  virtual void initialize(uint32_t w, uint32_t h);
  virtual void render(const Camera& camera);
  virtual void resize(uint32_t w, uint32_t h);

  Image2D* getImage() const { return image; };

private:
  Image2D* image;
  std::shared_ptr<ImageXYZC> _img;
};
