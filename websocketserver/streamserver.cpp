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
			//extract render mode
			int mode = json["mode"].toInt();

			//extract data set type
			QJsonValueRef dtype = json["datatype"];
			QJsonArray dtypes = dtype.toArray();
			QString type1 = dtypes[0].toString();
			QString type2 = dtypes[1].toString();


			//extract data channel
			QJsonValueRef chtype = json["datachannel"];
			int dataChannel = chtype.toInt();
			QString cell1 = "";
			QString cell2 = "";


			//extract animation state
			QJsonValueRef animstate = json["animationstate"];
			qreal crossfade = animstate.toDouble();
			//qDebug() << "animation state:" << crossfade;

			//messageobj.mouseDeltaRotate = { x:outX, y : outY };
			//messageobj.deltaRotate = { x:rotationAxis.x, y : rotationAxis.y, z : rotationAxis.z, angle : angle };
			QJsonValueRef deltaRotate = json["deltaRotate"];
			QJsonObject deltaRotate_obj = deltaRotate.toObject();
			QJsonValueRef mouseDeltaRotate = json["mouseDeltaRotate"];
			QJsonObject mouseDeltaRotate_obj = mouseDeltaRotate.toObject();
			int mseDx = mouseDeltaRotate_obj["x"].toInt();
			int mseDy = mouseDeltaRotate_obj["y"].toInt();
			//qDebug() << "MSE ROT " << mseDx << "," << mseDy;


			QJsonValueRef content = json["msgcontent"];
			QJsonObject content_obj = content.toObject();

			QJsonValueRef elements = content_obj["elements"];
			QJsonObject json2 = elements.toObject();

			QMatrix4x4 modelview = QMatrix4x4();
			modelview.setColumn(0, QVector4D(json2["0"].toDouble(), json2["1"].toDouble(), json2["2"].toDouble(), json2["3"].toDouble()));
			modelview.setColumn(1, QVector4D(json2["4"].toDouble(), json2["5"].toDouble(), json2["6"].toDouble(), json2["7"].toDouble()));
			modelview.setColumn(2, QVector4D(json2["8"].toDouble(), json2["9"].toDouble(), json2["10"].toDouble(), json2["11"].toDouble()));
			modelview.setColumn(3, QVector4D(json2["12"].toDouble(), json2["13"].toDouble(), json2["14"].toDouble(), json2["15"].toDouble()));

			//configure sub-component visibility
			bool structures[6]; // = {true, false, true, false, true, false};
			QJsonValueRef visibility = json["visibility"];
			QJsonObject visibility_obj = visibility.toObject();

			structures[0] = visibility_obj["0"].toBool();
			structures[1] = visibility_obj["1"].toBool();
			structures[2] = visibility_obj["2"].toBool();
			structures[3] = visibility_obj["3"].toBool();
			structures[4] = visibility_obj["4"].toBool();
			structures[5] = visibility_obj["5"].toBool();

			//encoding bool mask into integer
			int struc_asint = 0;
			for (int i = 0; i<6; ++i)
			{
				if (structures[i])
				{
					int mask = 0;
					mask = 1 << i;
					struc_asint = struc_asint | mask;
				}
			}

			//configure rendering
			double slider_settings[6];
			QJsonValueRef sliders = json["sliderset"];
			QJsonObject sliders_obj = sliders.toObject();

			slider_settings[0] = sliders_obj["0"].toDouble();
			slider_settings[1] = sliders_obj["1"].toDouble();
			slider_settings[2] = sliders_obj["2"].toDouble();
			slider_settings[3] = sliders_obj["3"].toDouble();
			slider_settings[4] = sliders_obj["4"].toDouble();
			slider_settings[5] = sliders_obj["5"].toDouble();

			//qDebug() << "slider settings: " << slider_settings[0];



			//parsing channel values:
			QJsonValueRef observed_json = json["observed"];
			QJsonArray observed = observed_json.toArray();

			QJsonValueRef modeled_json = json["modeled"];
			QJsonArray modeled = modeled_json.toArray();

			//            qDebug() << "observed: " << observed;
			//            qDebug() << "modeled: " << modeled;

			QList<QVariant> channelvalues;
			for (int i = 0; i<observed.size(); i++)
			{
				channelvalues.append(observed[i].toDouble());
			}
			for (int i = 0; i<modeled.size(); i++)
			{
				channelvalues.append(modeled[i].toDouble());
			}
			//*******************************


			channelvalues[0] = dataChannel;
			RenderParameters p(modelview, type1, cell1, type2, cell2, channelvalues, mode, crossfade, struc_asint, slider_settings[2], DEFAULT_IMAGE_FORMAT, 92, true);
			p.mseDx = mseDx;
			p.mseDy = mseDy;


			RenderRequest *request = new RenderRequest(pClient, p);
			this->getLeastBusyRenderer()->addRequest(request);


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

		RenderParameters p(b.getQueue());
		RenderRequest *request = new RenderRequest(pClient, p);
		this->getLeastBusyRenderer()->addRequest(request);

	}
}

void StreamServer::socketDisconnected()
{
	QWebSocket *pClient = qobject_cast<QWebSocket *>(sender());
	//if (m_debug)
	qDebug() << "socketDisconnected:" << pClient << "(" << pClient->closeCode() << ":" << pClient->closeReason() + ")";
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
		int v = request->getParameters().visibility;
		this->sampleCount[v]++;
		this->timings[v] = this->timings[v] + (request->getActualDuration() - this->timings[v]) / this->sampleCount[v];


		QString settings = QString::number(request->getParameters().visibility, 2);
		while (settings.length() < 8)
		{
			settings = "0" + settings;
		}

		QString fileName = "cache/" +
			settings + " - " +
			QString::number(rand()) + "." +
			QString(request->getParameters().format);

		qDebug() << "saving image to" << fileName;
		qDebug() << "(" << image.width() << "," << image.height() << ")";
		bool ok = image.save(fileName,
			request->getParameters().format,
			request->getParameters().quality);
		if (!ok) {
			qDebug() << "Could not save " << fileName;
		}
	}

	if (request->getClient() != 0)
	{
		QByteArray ba;
		QBuffer buffer(&ba);
		buffer.open(QIODevice::WriteOnly);
		image.save(&buffer, request->getParameters().format, request->getParameters().quality);

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
