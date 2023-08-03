#ifndef RENDERREQUEST_H
#define RENDERREQUEST_H

#include <QMatrix4x4>
#include <QWebSocket>

class Command;

class RenderRequest
{
public:
  RenderRequest(QWebSocket* client, std::vector<Command*> parameters, bool debug = false);
  ~RenderRequest();

  inline QWebSocket* getClient() { return client; }

  inline std::vector<Command*> getParameters() { return parameters; }

  inline int getDuration() { return this->estimatedDuration; }

  inline bool isDebug() { return debug; }

  inline void setActualDuration(qint64 actualDuration) { this->actualDuration = actualDuration; }

  inline qint64 getActualDuration() { return actualDuration; }

private:
  QWebSocket* client;
  std::vector<Command*> parameters;

  // an estimation of how many milliseconds will this request take to process
  int estimatedDuration;

  // how many nanoseconds did it actually take
  qint64 actualDuration;

  bool debug;
};

#endif // RENDERREQUEST_H
