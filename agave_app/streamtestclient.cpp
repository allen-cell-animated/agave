#include "streamtestclient.h"

#include "commandBuffer.h"
#include "renderlib/command.h"

StreamTestClient::StreamTestClient(const QUrl& url, bool debug, QObject* parent)
  : QObject(parent)
  , m_url(url)
  , m_debug(debug)
{
  if (m_debug)
    qDebug() << "Testclient WebSocket server:" << url;
  connect(&m_webSocket, &QWebSocket::connected, this, &StreamTestClient::onConnected);
  connect(&m_webSocket, &QWebSocket::disconnected, this, &StreamTestClient::closed);
  m_webSocket.open(QUrl(url));
}

void
StreamTestClient::onConnected()
{
  if (m_debug)
    qDebug() << "Testclient WebSocket connected";
  connect(&m_webSocket, &QWebSocket::textMessageReceived, this, &StreamTestClient::onTextMessageReceived);
  connect(&m_webSocket, &QWebSocket::binaryMessageReceived, this, &StreamTestClient::onBinaryMessageReceived);
  m_webSocket.sendTextMessage(QStringLiteral("Hello, world!"));

  // build up some binary commands to send.
  std::vector<Command*> commands;
  commands.push_back(new SetDiffuseColorCommand(SetDiffuseColorCommandD{ 0, 1.0, 1.0, 1.0, 1.0 }));

  createCommandBuffer(commands);
}

void
StreamTestClient::onTextMessageReceived(const QString& message)
{
  if (m_debug)
    qDebug() << "Testclient Message received:" << message;
}

void
StreamTestClient::onBinaryMessageReceived(const QByteArray& message)
{
  if (m_debug)
    qDebug() << "Testclient Binary Message received:" << message;
}

void
StreamTestClient::closed()
{
  qDebug() << "Testclient WebSocket closed";
}

void
StreamTestClient::createCommandBuffer(const std::vector<Command*>& commands)
{
  QByteArray buffer;

  commandBuffer* cb = commandBuffer::createBuffer(commands);
  // fill buffer with data from cb
}
