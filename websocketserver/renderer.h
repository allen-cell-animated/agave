#ifndef RENDERER_H
#define RENDERER_H

#include <QObject>
#include <QList>

#include <QMutex>
#include <QOpenGLTexture>
#include <QOpenGLContext>
#include <QOffscreenSurface>
#include <QThread>

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

signals:
	void kill();
	void requestProcessed(RenderRequest *request, QImage img);
};

#endif // RENDERER_H
