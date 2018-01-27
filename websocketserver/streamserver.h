#ifndef STREAMSERVER_H
#define STREAMSERVER_H

#include <QtCore/QObject>
#include <QtCore/QList>
#include <QtCore/QByteArray>

#include <QPixmap>
#include <QImage>
#include <QByteArray>
#include <QBuffer>
#include <QWidget>
#include <QtDebug>
#include <QApplication>
#include <QMouseEvent>
#include <QWheelEvent>

#include <QSslError>

#include "renderer.h"

QT_FORWARD_DECLARE_CLASS(QWebSocketServer)
QT_FORWARD_DECLARE_CLASS(QWebSocket)

#define THREAD_COUNT 1

class StreamServer : public QObject
{
	Q_OBJECT
public:
	explicit StreamServer(quint16 port, bool debug, QObject *parent = Q_NULLPTR);
	~StreamServer();

	inline int getClientsCount()
	{
		return clients.count();
	}

	inline QList<QWebSocket *> getClients()
	{
		return clients;
	}

	inline int getThreadsCount()
	{
		return THREAD_COUNT;
	}

	inline QList<int> getThreadsLoad()
	{
		QList<int> loads;
		foreach(Renderer *renderer, this->renderers)
		{
			loads << renderer->getTotalQueueDuration();
		}

		return loads;
	}

	inline QList<int> getThreadsRequestCount()
	{
		QList<int> requests;
		foreach(Renderer *renderer, this->renderers)
		{
			requests << renderer->getRequestCount();
		}

		return requests;
	}


signals:
	void closed();

	private slots:
	void onNewConnection();
	void onSslErrors(const QList<QSslError> &errors);
	void processTextMessage(QString message);
	void processBinaryMessage(QByteArray message);
	void socketDisconnected();
	void sendImage(RenderRequest *request, QImage image);

private:
	//QVector<int> sampleCount;
	//QVector<qreal> timings;

	Renderer *getLeastBusyRenderer();

	QWebSocketServer *webSocketServer;
	QList<QWebSocket *> clients;
	bool debug;

	QList<Renderer *> renderers;

	QByteArray previousArray;
};

#endif //STREAMSERVER_H
