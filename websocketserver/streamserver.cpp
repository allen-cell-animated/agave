#include "streamserver.h"
#include "QtWebSockets/qwebsocketserver.h"
#include "QtWebSockets/qwebsocket.h"
#include <QtCore/QDebug>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSslConfiguration>
#include <QSslCertificate>
#include <QSslKey>

#include "commandBuffer.h"
#include "util.h"

QT_USE_NAMESPACE

const char *DEFAULT_IMAGE_FORMAT = "JPG";

StreamServer::StreamServer(quint16 port, bool debug, QObject *parent) :
	QObject(parent),
	webSocketServer(new QWebSocketServer(QStringLiteral("Marion"), QWebSocketServer::NonSecureMode, this)),
	clients(),
	debug(debug)
{
	connect(this, &StreamServer::closed, qApp, &QApplication::quit);

	qDebug() << "Server is starting up with" << THREAD_COUNT << "threads, listening on port" << port << "...";

	// assumption : each renderer is rendering the same scene, just possibly for different clients.
	
	for (int i = 0; i < THREAD_COUNT; i++)
	{
		this->renderers << new Renderer("Thread " + QString::number(i), this);
		connect(this->renderers.last(), SIGNAL(requestProcessed(RenderRequest*, QImage)), this, SLOT(sendImage(RenderRequest*, QImage)));

		qDebug() << "Starting thread" << i << "...";
		this->renderers.last()->start();
	}
	qDebug() << "Done.";

	qDebug() << "Sampling the rendering parameters...";

	int settingsCount = 64;

	this->timings.resize(settingsCount);
	this->sampleCount.resize(settingsCount);

#if 0
	for (int i = 0; i < 10; i++)
	{
		//sample the surface of the sphere
		qreal u = Util::rrandom();
		qreal v = Util::rrandom();
		qreal th = 2.0 * PI * u;
		qreal ph = acos(2.0 * v - 1.0);

		qreal r = 1.0 /*+ Util::rrandom() * 1.0*/;

		//cartesian
		qreal x = r * cos(th) * sin(ph);
		qreal y = r * sin(th) * cos(ph);
		qreal z = r * cos(ph);

		QMatrix4x4 modelview;
		modelview.lookAt(QVector3D(x, y, z), QVector3D(0.0, 0.0, 0.0), QVector3D(0.0, 1.0, 0.0));

		int settings = rand() % 64;
		//issue the request

		//*******************************
		//dummy parameters for testing
		QString type1 = "Interphase_5cells";
		QString type2 = "Mitotic_2cells";
		QString cell1 = "20161216_C02_005_6";
		QString cell2 = "20160705_S03_058_7";
		QList<QVariant> channelvalues;
		for (int i = 0; i<13; i++)
		{
			channelvalues.append(0.5);
		}
		qreal crossfade = 0.0;
		int mode = 0;
		//*******************************


		RenderParameters p(modelview, type1, cell1, type2, cell2, channelvalues, mode, crossfade, settings, 0.0, DEFAULT_IMAGE_FORMAT, 92, true);
		//RenderParameters p(modelview, settings, 0.0, DEFAULT_IMAGE_FORMAT, 92);

		RenderRequest *request = new RenderRequest(0, p, false);
		this->getLeastBusyRenderer()->addRequest(request);
	}

	for (int settings = 0; settings < settingsCount; settings++)
	{
		this->timings[settings] = 0;
		this->sampleCount[settings] = 0;

		for (int i = 0; i < 1; i++)
		{
			//sample the surface of the sphere
			qreal u = Util::rrandom();
			qreal v = Util::rrandom();
			qreal th = 2.0 * PI * u;
			qreal ph = acos(2.0 * v - 1.0);

			qreal r = 1.0 /*+ Util::rrandom() * 1.0*/;

			//cartesian
			qreal x = r * cos(th) * sin(ph);
			qreal y = r * sin(th) * cos(ph);
			qreal z = r * cos(ph);

			QMatrix4x4 modelview;
			modelview.lookAt(QVector3D(x, y, z), QVector3D(0.0, 0.0, 0.0), QVector3D(0.0, 1.0, 0.0));


			//issue the request

			//*******************************
			//dummy parameters for testing
			QString type1 = "Interphase_5cells";
			QString type2 = "Mitotic_2cells";
			QString cell1 = "20161216_C02_005_6";
			QString cell2 = "20160705_S03_058_7";
			QList<QVariant> channelvalues;
			for (int i = 0; i<13; i++)
			{
				channelvalues.append(0.5);
			}
			qreal crossfade = 0.0;
			int mode = 0;
			//*******************************


			RenderParameters p(modelview, type1, cell1, type2, cell2, channelvalues, mode, crossfade, settings, 0.0, DEFAULT_IMAGE_FORMAT, 92, true);
			//RenderParameters p(modelview, settings, 0.0, DEFAULT_IMAGE_FORMAT, 92);

			RenderRequest *request = new RenderRequest(0, p, true);
			this->getLeastBusyRenderer()->addRequest(request);
		}
	}
#endif
	qDebug() << "Done.";

	QSslConfiguration sslConfiguration;
	QFile certFile(QStringLiteral("mr.crt"));
	QFile keyFile(QStringLiteral("mr.key"));
	certFile.open(QIODevice::ReadOnly);
	keyFile.open(QIODevice::ReadOnly);
	QSslCertificate certificate(&certFile, QSsl::Pem);
	QSslKey sslKey(&keyFile, QSsl::Rsa, QSsl::Pem);
	certFile.close();
	keyFile.close();
	sslConfiguration.setPeerVerifyMode(QSslSocket::VerifyNone);
	sslConfiguration.setLocalCertificate(certificate);
	sslConfiguration.setPrivateKey(sslKey);
	sslConfiguration.setProtocol(QSsl::TlsV1SslV3);
	webSocketServer->setSslConfiguration(sslConfiguration);

	qDebug() << "QSslSocket::supportsSsl() = " << QSslSocket::supportsSsl();

	if (webSocketServer->listen(QHostAddress::Any, port))
	{
		if (debug)
			qDebug() << "Streamserver listening on port" << port;
		connect(webSocketServer, &QWebSocketServer::newConnection, this, &StreamServer::onNewConnection);
		connect(webSocketServer, &QWebSocketServer::closed, this, &StreamServer::closed);
		connect(webSocketServer, &QWebSocketServer::sslErrors, this, &StreamServer::onSslErrors);
	}

	qDebug() << "Server initialization done.";
}

StreamServer::~StreamServer()
{
	webSocketServer->close();
	qDeleteAll(clients.begin(), clients.end());
}

void StreamServer::onSslErrors(const QList<QSslError> &errors)
{
	foreach(QSslError error, errors)
	{
		qDebug() << "SSL ERROR: " << error.errorString();
	}
}

void StreamServer::onNewConnection()
{
	// fire up new renderer?

	QWebSocket *pSocket = webSocketServer->nextPendingConnection();

	connect(pSocket, &QWebSocket::textMessageReceived, this, &StreamServer::processTextMessage);
	connect(pSocket, &QWebSocket::binaryMessageReceived, this, &StreamServer::processBinaryMessage);
	connect(pSocket, &QWebSocket::disconnected, this, &StreamServer::socketDisconnected);

	clients << pSocket;

	//if (m_debug)
	qDebug() << "new client!" << pSocket->resourceName() << "; " << pSocket->peerAddress().toString() << ":" << pSocket->peerPort() << "; " << pSocket->peerName();
}

Renderer *StreamServer::getLeastBusyRenderer()
{
	Renderer *leastBusy = 0;
	foreach(Renderer *renderer, this->renderers)
	{
		if (leastBusy == 0 || renderer->getTotalQueueDuration() < leastBusy->getTotalQueueDuration())
		{
			leastBusy = renderer;
		}
	}

	return leastBusy;
}

void StreamServer::processTextMessage(QString message)
{
	QWebSocket *pClient = qobject_cast<QWebSocket *>(sender());

	if (debug)
		qDebug() << "Message received:" << message;

	if (pClient)
	{
		QJsonDocument doc = QJsonDocument::fromJson(message.toUtf8());

		// Get JSON object
		QJsonObject json = doc.object();

		int msgtype = json["msgtype"].toInt();
		//qDebug() << "Message type:" << msgtype;

		switch (msgtype)
		{
		case 0:
		{
			//RenderRequest *request = new RenderRequest(pClient, p);
			//this->getLeastBusyRenderer()->addRequest(request);
			break;
		}
		case 1:
		{
#if 0
			QString json = this->getLeastBusyRenderer()->getMarion()->library("cellloader")->getInterface()->getValue("multichannel.filelist").toString();
			//qDebug() << "the json: " << json;

			if (pClient != 0)
			{
				pClient->sendTextMessage(json);
			}

			//qDebug() << "json:" << json;
#endif
			break;
		}
		case 2:
		{
			break;
		}

		//default:
		//break;
		}
	}
}

void StreamServer::processBinaryMessage(QByteArray message)
{
	QWebSocket *pClient = qobject_cast<QWebSocket *>(sender());
//	if (debug)
//		qDebug() << "Binary Message received:" << message;
	if (pClient)
	{
		// the message had better be an encoded command stream.  check a header perhaps?

		commandBuffer b(message.length(), reinterpret_cast<const uint8_t*>(message.constData()));
		b.processBuffer();

		// one message is a list of commands to run before rendering.
		// the complete message amounts to a single render request and an image is expected to come out of it.
		RenderRequest *request = new RenderRequest(pClient, b.getQueue());
		this->getLeastBusyRenderer()->addRequest(request);
	}
}

void StreamServer::socketDisconnected()
{
	QWebSocket *pClient = qobject_cast<QWebSocket *>(sender());
	//if (m_debug)
	//qDebug() << "new client!" << pSocket->resourceName() << "; " << pSocket->peerAddress().toString() << ":" << pSocket->peerPort() << "; " << pSocket->peerName();
	qDebug() << "socketDisconnected:" << pClient->resourceName() << "(" << pClient->closeCode() << ":" << pClient->closeReason() + ")";
	if (pClient) {
		clients.removeAll(pClient);
		pClient->deleteLater();
	}
}

void StreamServer::sendImage(RenderRequest *request, QImage image)
{
	//qDebug() << "sending image to" << request->getClient();

	//QImage image = widget.grab().toImage();

	//TODO: why does this work at all??

	/*Canvas *canvas = dynamic_cast<Canvas*> (widget);
	QImage image;
	image = canvas->getImage();*/

	/*QString cameraString = QString::number(canvas->getRotation().x(), 'f',1) + ";" +
	QString::number(canvas->getRotation().y(), 'f',1) + ";" +
	QString::number(canvas->getRotation().z(), 'f',1) + ";" +
	QString::number(canvas->getRotation().scalar(), 'f',1) + ";";

	QString hash = QString(QCryptographicHash::hash(cameraString.toUtf8(), QCryptographicHash::Md5).toHex());*/

	//qDebug() << cameraString;
	//qDebug() << hash;

	if (request->isDebug())
	{
		if (image.isNull()) {
			qDebug() << "NULL IMAGE RECEIVED AT STREAMSERVER FROM RENDERER";
		}

		//running mean
		int v = 0;
		this->sampleCount[v]++;
		this->timings[v] = this->timings[v] + (request->getActualDuration() - this->timings[v]) / this->sampleCount[v];

		QString fileName = "cache/" +
			QString::number(rand()) + "." +
			QString(DEFAULT_IMAGE_FORMAT);

		qDebug() << "saving image to" << fileName;
		qDebug() << "(" << image.width() << "," << image.height() << ")";
		bool ok = image.save(fileName,
			DEFAULT_IMAGE_FORMAT,
			92);
		if (!ok) {
			qDebug() << "Could not save " << fileName;
		}
	}

	if (request->getClient() != 0)
	{
		QByteArray ba;
		QBuffer buffer(&ba);
		buffer.open(QIODevice::WriteOnly);
		image.save(&buffer, DEFAULT_IMAGE_FORMAT, 92);

		request->getClient()->sendBinaryMessage(ba);
	}


	/*QString fileName = "cache/" + hash + "." + QString(format);

	QFileInfo fi(fileName);*/

	/*if (!fi.exists())
	{
	image.save(fileName, DEFAULT_IMAGE_FORMAT, 1);
	}*/
	/*else
	{
	QImage loaded(fileName);

	QBuffer buffer(&ba);
	buffer.open(QIODevice::WriteOnly);
	loaded.save(&buffer, format, quality);
	}*/


	/*if (ba != previousArray || true)
	{
	client->sendBinaryMessage(ba);
	}

	previousArray = QByteArray(ba);*/
}
