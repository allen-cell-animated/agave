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
#include "renderlib/Logging.h"

QT_USE_NAMESPACE

const char *DEFAULT_IMAGE_FORMAT = "JPG";

void StreamServer::createNewRenderer(QWebSocket* client) {
	int i = this->_renderers.length();
	Renderer* r = new Renderer("Thread " + QString::number(i), this, _openGLMutex);
	this->_renderers << r;
	connect(r, SIGNAL(requestProcessed(RenderRequest*, QImage)), this, SLOT(sendImage(RenderRequest*, QImage)));
	connect(r, SIGNAL(sendString(RenderRequest*, QString)), this, SLOT(sendString(RenderRequest*, QString)));

	qDebug() << "Starting thread" << i << "...";
	r->start();

	_clientRenderers[client] = r;
}

StreamServer::StreamServer(quint16 port, bool debug, QObject *parent) :
	QObject(parent),
	_webSocketServer(new QWebSocketServer(QStringLiteral("Marion"), QWebSocketServer::NonSecureMode, this)),
	_clients(),
	_renderers(),
	debug(debug)
{
	connect(this, &StreamServer::closed, qApp, &QApplication::quit);

	qDebug() << "Server is starting up with" << THREAD_COUNT << " max threads, listening on port" << port << "...";

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
	_webSocketServer->setSslConfiguration(sslConfiguration);

	qDebug() << "QSslSocket::supportsSsl() = " << QSslSocket::supportsSsl();

	if (_webSocketServer->listen(QHostAddress::Any, port))
	{
		if (debug)
			qDebug() << "Streamserver listening on port" << port;
		connect(_webSocketServer, &QWebSocketServer::newConnection, this, &StreamServer::onNewConnection);
		connect(_webSocketServer, &QWebSocketServer::closed, this, &StreamServer::closed);
		connect(_webSocketServer, &QWebSocketServer::sslErrors, this, &StreamServer::onSslErrors);
	}

	qDebug() << "Server initialization done.";
}

StreamServer::~StreamServer()
{
	_webSocketServer->close();
	qDeleteAll(_clients.begin(), _clients.end());
	qDeleteAll(_renderers.begin(), _renderers.end());
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
	QWebSocket *pSocket = _webSocketServer->nextPendingConnection();

	// fire up new renderer?
	if (_renderers.length() >= THREAD_COUNT) {
		// handle this better.
		pSocket->abort();
		LOG_DEBUG << "TOO MANY CONNECTIONS.";
		return;
	}


	connect(pSocket, &QWebSocket::textMessageReceived, this, &StreamServer::processTextMessage);
	connect(pSocket, &QWebSocket::binaryMessageReceived, this, &StreamServer::processBinaryMessage);
	connect(pSocket, &QWebSocket::disconnected, this, &StreamServer::socketDisconnected);
//	QObject::connect(pSocket, &QWebSocket::error, [pSocket](QAbstractSocket::SocketError e)
//	{
//		// Handle error here...
//		qDebug() << pSocket->errorString();
//	});

	_clients << pSocket;
	createNewRenderer(pSocket);

	//if (m_debug)
	LOG_DEBUG << "new client!" << pSocket->resourceName().toStdString() << "; " << pSocket->peerAddress().toString().toStdString() << ":" << pSocket->peerPort() << "; " << pSocket->peerName().toStdString();
}

Renderer* StreamServer::getRendererForClient(QWebSocket* client) {
	return _clientRenderers[client];
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

		// hand the commands over to the RenderRequest. 
		// RenderRequest will assume ownership and delete them

		RenderRequest *request = new RenderRequest(pClient, b.getQueue());
		this->getRendererForClient(pClient)->addRequest(request);
	}
}

void StreamServer::socketDisconnected()
{
	QWebSocket *pClient = qobject_cast<QWebSocket *>(sender());
	//if (m_debug)
	//qDebug() << "new client!" << pSocket->resourceName() << "; " << pSocket->peerAddress().toString() << ":" << pSocket->peerPort() << "; " << pSocket->peerName();
	qDebug() << "socketDisconnected:" << pClient->resourceName() << "(" << pClient->closeCode() << ":" << pClient->closeReason() + ")";
	if (pClient) {
		Renderer* r = getRendererForClient(pClient);
		QObject::connect(r, &Renderer::finished, r, &QObject::deleteLater);
		if (r) {
			r->requestInterruption();
		}
		_clients.removeAll(pClient);
		_renderers.removeAll(r);
		_clientRenderers.remove(pClient);
		pClient->deleteLater();
	}
}

void StreamServer::sendImage(RenderRequest *request, QImage image)
{
	if (request->isDebug())
	{
		if (image.isNull()) {
			qDebug() << "NULL IMAGE RECEIVED AT STREAMSERVER FROM RENDERER";
		}
	}

	QWebSocket* client = request->getClient();
	if (client != 0 && _clients.contains(client) && client->isValid() && client->state() == QAbstractSocket::ConnectedState)
	{
		QByteArray ba;
		QBuffer buffer(&ba);
		buffer.open(QIODevice::WriteOnly);
		image.save(&buffer, DEFAULT_IMAGE_FORMAT, 92);

		client->sendBinaryMessage(ba);
	}

	// this is the end of the line for a request.
	delete request;
}

void StreamServer::sendString(RenderRequest *request, QString s)
{
	QWebSocket* client = request->getClient();
	if (client != 0 && _clients.contains(client) && client->isValid() && client->state() == QAbstractSocket::ConnectedState)
	{
		client->sendTextMessage(s);
	}
	// do not dispose of request here. 
	// see requestProcessed<-->sendImage from Renderer.
}
