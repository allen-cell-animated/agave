#pragma once
#include "IRenderWindow.h"
#include "AppScene.h"
#include "Status.h"
#include "Timing.h"

#include <QElapsedTimer>

#include <memory>

class CScene;
class Image2D;
class Image3Dv33;
class ImageXYZC;

class RenderGL :
	public IRenderWindow
{
public:
	RenderGL(CScene* scene);
	virtual ~RenderGL();

	virtual void initialize(uint32_t w, uint32_t h);
	virtual void render(const Camera& camera);
	virtual void resize(uint32_t w, uint32_t h);

	virtual CStatus* getStatusInterface() { return &_status; }
	virtual RenderParams& renderParams();
	virtual Scene& scene();

	Image3Dv33* getImage() const { return image3d; };
	void setImage(std::shared_ptr<ImageXYZC> img);
private:
	Image3Dv33 *image3d;
	CScene* _scene;

	Scene _appScene;
	RenderParams _renderParams;

	CStatus _status;
	CTiming _timingRender;
	QElapsedTimer _timer;

	int _w, _h;
};

