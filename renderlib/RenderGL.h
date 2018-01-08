#pragma once
#include "IRenderWindow.h"

#include <memory>

class CScene;
class Image2D;
class Image3Dv33;
class ImageXYZC;

class RenderGL :
	public IRenderWindow
{
public:
	RenderGL(std::shared_ptr<ImageXYZC>  img, CScene* scene);
	virtual ~RenderGL();

	virtual void initialize(uint32_t w, uint32_t h);
	virtual void render(const Camera& camera);
	virtual void resize(uint32_t w, uint32_t h);

	Image3Dv33* getImage() const { return image3d; };
	void setImage(std::shared_ptr<ImageXYZC> img);
private:
	Image3Dv33 *image3d;
	std::shared_ptr<ImageXYZC>  _img;
	CScene* _scene;
};

