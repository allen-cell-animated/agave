#pragma once

#include <QObject>
#include <QUrl>
#include <QtWebSockets/QWebSocket>

class Command;

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