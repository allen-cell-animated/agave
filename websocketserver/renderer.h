#ifndef RENDERER_H
#define RENDERER_H

#include <QObject>
#include <QList>

#include <QOpenGLTexture>
#include <QOpenGLContext>
#include <QOffscreenSurface>
#include <QMutex>
#include <QThread>

#include <memory>

class ImageXYZC;
class RenderGLCuda;
class CScene;
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

	inline int getTotalQueueDuration()
	{
		return this->totalQueueDuration;
	}

	inline int getRequestCount()
	{
		return this->requests.count();
	}

//	inline Marion *getMarion()
//	{
//		return this->marion;
//	}

protected:
	QString id;

	QImage render(RenderParameters p);

	void resizeGL(int internalWidth, int internalHeight);
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

	//Marion *marion;

	int frameNumber;
	QTime time;

	void renderScene(QString scene);
	void displayScene(QString scene);

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

	QMatrix4x4 projection;
	QMatrix4x4 modelview;

	void myVolumeInit();
	struct myVolumeData {
		std::shared_ptr<ImageXYZC> _image;
		CScene* _scene;
		RenderGLCuda* _renderer;

		myVolumeData() : _scene(nullptr), _renderer(nullptr) {}
	} myVolumeData;

signals:
	void kill();
	void requestProcessed(RenderRequest *request, QImage img);
};

#endif // RENDERER_H
