#include "streamtestclient.h"

#include "commandBuffer.h"
#include "renderlib/command.h"

#include <QFile>

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

  // build up some binary commands to send.
  std::vector<Command*> commands;
  commands.push_back(new LoadVolumeFromFileCommand({ "E:\\data\\caax-8.ome.tiff", 0, 63 }));
  commands.push_back(new SetResolutionCommand({ 681, 612 }));
  commands.push_back(new SetBackgroundColorCommand({ 0.0, 0.0, 0.0 }));
  commands.push_back(new ShowBoundingBoxCommand({ 1 }));
  commands.push_back(new SetBoundingBoxColorCommand({ 1.0, 1.0, 1.0 }));
  commands.push_back(new SetRenderIterationsCommand({ 128 }));
  commands.push_back(new SetPrimaryRayStepSizeCommand({ 4.0 }));
  commands.push_back(new SetSecondaryRayStepSizeCommand({ 4.0 }));
  commands.push_back(new SetVoxelScaleCommand({ 0.541667, 0.541667, 2 }));
  commands.push_back(new SetClipRegionCommand({ 0, 1, 0, 1, 0, 1 }));
  commands.push_back(new SetCameraPosCommand({ 0.5, 0.342181, 2.03729 }));
  commands.push_back(new SetCameraTargetCommand({ 0.5, 0.342181, 0.0372853 }));
  commands.push_back(new SetCameraUpCommand({ 0, 1, 0 }));
  commands.push_back(new SetCameraProjectionCommand({ 1, 0.5 }));
  commands.push_back(new SetCameraExposureCommand({ 0.8065 }));
  commands.push_back(new SetDensityCommand({ 41.0965 }));
  commands.push_back(new SetCameraApertureCommand({ 0.0 }));
  commands.push_back(new SetCameraFocalDistanceCommand({ 0.75 }));
  commands.push_back(new EnableChannelCommand({ 0, 0 }));
  commands.push_back(new SetDiffuseColorCommand({ 0, 1, 0, 1, 1 }));
  commands.push_back(new SetSpecularColorCommand({ 0, 0, 0, 0, 0 }));
  commands.push_back(new SetEmissiveColorCommand({ 0, 0, 0, 0, 0 }));
  commands.push_back(new SetGlossinessCommand({ 0, 1 }));
  commands.push_back(new SetOpacityCommand({ 0, 1 }));
  commands.push_back(new SetPercentileThresholdCommand({ 0, 0.5, 0.98 }));
  commands.push_back(new EnableChannelCommand({ 1, 1 }));
  commands.push_back(new SetDiffuseColorCommand({ 1, 1, 1, 1, 1 }));
  commands.push_back(new SetSpecularColorCommand({ 1, 0, 0, 0, 0 }));
  commands.push_back(new SetEmissiveColorCommand({ 1, 0, 0, 0, 0 }));
  commands.push_back(new SetGlossinessCommand({ 1, 1 }));
  commands.push_back(new SetOpacityCommand({ 1, 1 }));
  commands.push_back(new SetPercentileThresholdCommand({ 1, 0.6023, 0.98 }));
  commands.push_back(new SetSkylightTopColorCommand({ 1, 1, 1 }));
  commands.push_back(new SetSkylightMiddleColorCommand({ 1, 1, 1 }));
  commands.push_back(new SetSkylightBottomColorCommand({ 1, 1, 1 }));
  commands.push_back(new SetLightPosCommand({ 0, 10, 0.2693, 0.7629 }));
  commands.push_back(new SetLightColorCommand({ 0, 166.667, 333.333, 500 }));
  commands.push_back(new SetLightSizeCommand({ 0, 1, 1 }));
  commands.push_back(new SessionCommand({ "caax-8.png" }));
  commands.push_back(new OrbitCameraCommand({ 60, 0 }));
  // r.batch_render_turntable(number_of_frames = 45, output_name = "orbitz")
  //#r.redraw()
  commands.push_back(new RequestRedrawCommand({}));
  commandBuffer* cb = commandBuffer::createBuffer(commands);

  m_webSocket.sendBinaryMessage(QByteArray((const char*)cb->head(), cb->length()));

  // clean up
  delete cb;
  for (auto c : commands) {
    delete c;
  }
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
  //if (m_debug)
  //  qDebug() << "Testclient Binary Message received:" << message;

  // a single binary message is assumed to be a serialized png file ready to save.
  QFile newDoc("C:\\Users\\danielt\\Desktop\\test.png");
  if (newDoc.open(QIODevice::WriteOnly)) {
    newDoc.write(message);
  }

  newDoc.close();
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
