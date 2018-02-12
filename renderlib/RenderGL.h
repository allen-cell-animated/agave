#pragma once
#include "IRenderWindow.h"
#include "AppScene.h"
#include "Status.h"
#include "Timing.h"

#include <QElapsedTimer>

#include <memory>

class Image3Dv33;
class ImageXYZC;
class RenderSettings;

class RenderGL :
	public IRenderWindow
{
public:
	RenderGL(RenderSettings* rs);
	virtual ~RenderGL();

	virtual void initialize(uint32_t w, uint32_t h);
	virtual void render(const CCamera& camera);
	virtual void resize(uint32_t w, uint32_t h);
	virtual void cleanUpResources();

	virtual CStatus* getStatusInterface() { return &_status; }
	virtual RenderParams& renderParams();
	virtual Scene* scene();
	virtual void setScene(Scene* s);

	Image3Dv33* getImage() const { return image3d; };
private:
	Image3Dv33 *image3d;
	RenderSettings* _renderSettings;

	Scene* _scene;
	RenderParams _renderParams;

	CStatus _status;
	CTiming _timingRender;
	QElapsedTimer _timer;

	int _w, _h;

	void initFromScene();
};
