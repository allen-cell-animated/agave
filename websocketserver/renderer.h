#ifndef RENDERER_H
#define RENDERER_H

#include <QObject>
#include <QList>

#include <QOpenGLFramebufferObject>
#include <QOpenGLTexture>
#include <QOpenGLContext>
#include <QOffscreenSurface>
#include <QMutex>
#include <QThread>

#include <memory>

class commandBuffer;
class CCamera;
class ImageXYZC;
class RenderGLCuda;
class RenderSettings;
class Scene;

//#include "dynamiclibrary.h"

//#include "marion.h"
//#include "plugins.h"

#include "renderrequest.h"


class Renderer : public QThread
{
	Q_OBJECT

public:
	Renderer(QString id, QObject *parent = 0);
	~Renderer();

	void init();
	void run();

	void addRequest(RenderRequest *request);
	bool processRequest();
	void processCommandBuffer(std::vector<Command*>& cmds);

	inline int getTotalQueueDuration()
	{
		return this->totalQueueDuration;
	}

	inline int getRequestCount()
	{
		return this->requests.count();
	}

	// 1 = continuous re-render, 0 = only wait for redraw commands
	void setStreamMode(int32_t mode) { _streamMode = mode; }

	void resizeGL(int internalWidth, int internalHeight);

protected:
	QString id;

	QImage render();

	void reset(int from = 0);

	int getTime();

	QList<RenderRequest *> requests;
	int totalQueueDuration;

	void shutDown();

private:
	QMutex mutex;

	QOpenGLContext *context;
	QOffscreenSurface *surface;
	QOpenGLFramebufferObject *fbo;

	int32_t _streamMode;
	int32_t _width, _height;

	int frameNumber;
	QTime time;

	class SceneDescription
	{
	public:
		inline SceneDescription(QString name, int start, int end) :
			name(name),
			start(start),
			end(end)
		{}

		QString name;
		int start;
		int end;
	};

	QList<SceneDescription> scenes;

	// TODO move this info.  This class only knows about some abstract renderer and a scene object.
	void myVolumeInit();
	struct myVolumeData {
		RenderSettings* _renderSettings;
		RenderGLCuda* _renderer;
		Scene* _scene;
		CCamera* _camera;

		myVolumeData() : _camera(nullptr), _scene(nullptr), _renderSettings(nullptr), _renderer(nullptr) {}
	} myVolumeData;

signals:
	void kill();
	void requestProcessed(RenderRequest *request, QImage img);
};

#endif // RENDERER_H
