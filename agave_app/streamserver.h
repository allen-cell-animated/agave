#ifndef STREAMSERVER_H
#define STREAMSERVER_H

#include <QApplication>
#include <QBuffer>
#include <QByteArray>
#include <QImage>
#include <QList>
#include <QMouseEvent>
#include <QObject>
#include <QPixmap>
#include <QWheelEvent>
#include <QWidget>
#include <QtDebug>

#include <QSslError>

#include "renderer.h"

QT_FORWARD_DECLARE_CLASS(QWebSocketServer)
QT_FORWARD_DECLARE_CLASS(QWebSocket)

#define THREAD_COUNT 4

class StreamServer : public QObject
{
  Q_OBJECT
public:
  explicit StreamServer(quint16 port, bool debug, QObject* parent = Q_NULLPTR);
  ~StreamServer();

  inline int getClientsCount() { return _clients.count(); }

  inline QList<QWebSocket*> getClients() { return _clients; }

  inline int getThreadsCount() { return _renderers.length(); }

  inline QList<int> getThreadsLoad()
  {
    QList<int> loads;
    foreach (Renderer* renderer, this->_renderers) {
      loads << renderer->getTotalQueueDuration();
    }

    return loads;
  }

  inline QList<int> getThreadsRequestCount()
  {
    QList<int> requests;
    foreach (Renderer* renderer, this->_renderers) {
      requests << renderer->getRequestCount();
    }

    return requests;
  }

signals:
  void closed();

private slots:
  void onNewConnection();
  void onSslErrors(const QList<QSslError>& errors);
  void processTextMessage(QString message);
  void processBinaryMessage(QByteArray message);
  void socketDisconnected();
  void sendImage(RenderRequest* request, QImage image);
  void sendString(RenderRequest* request, QString s);

private:
  // Renderer *getLeastBusyRenderer();
  Renderer* getRendererForClient(QWebSocket* client);

  QWebSocketServer* _webSocketServer;

  QList<QWebSocket*> _clients;
  QList<Renderer*> _renderers;
  QMap<QWebSocket*, Renderer*> _clientRenderers;

  bool debug;

  void createNewRenderer(QWebSocket* client);

  QMutex _openGLMutex;
};

#endif // STREAMSERVER_H
