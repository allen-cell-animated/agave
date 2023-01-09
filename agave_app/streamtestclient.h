#pragma once

#include <QObject>
#include <QUrl>
#include <QtWebSockets/QWebSocket>

class Command;

// to be used in stream mode to send commands in-process to the server.
// commands will be binary encoded and decoded on the server side.
class StreamTestClient : public QObject
{
  Q_OBJECT

public:
  StreamTestClient(const QUrl& url, bool debug, QObject* parent = nullptr);

public slots:
  void onTextMessageReceived(const QString& message);
  void onBinaryMessageReceived(const QByteArray& message);
  void onConnected();
  void closed();

private:
  QUrl m_url;
  bool m_debug;
  QWebSocket m_webSocket;

  void createCommandBuffer(const std::vector<Command*>& commands);
};