#ifndef RENDERER_H
#define RENDERER_H

#include "glad/glad.h"

#include "renderlib/command.h"
#include "renderlib/gesture/gesture.h"
#include "renderlib/graphics/gl/Util.h"
#include "renderlib/graphics/GestureGraphicsGL.h"
#include "renderlib/io/FileReader.h"
#include "renderlib/renderlib.h"
#include "renderrequest.h"

#include <QElapsedTimer>
#include <QList>
#include <QMutex>
#include <QObject>
#include <QOpenGLContext>
#include <QThread>
#include <QWaitCondition>

#include <memory>

class commandBuffer;
class CCamera;
class ImageXYZC;
class IRenderWindow;
class RenderSettings;
class Scene;

class Renderer
  : public QThread
  , public RendererCommandInterface
{
  Q_OBJECT

public:
  Renderer(QString id, QObject* parent, QMutex& mutex);
  virtual ~Renderer();

  void configure(IRenderWindow* renderer,
                 const RenderSettings& renderSettings,
                 const Scene& scene,
                 const CCamera& camera,
                 const LoadSpec& loadSpec,
                 // rendererMode ignored if renderer is non-null
                 renderlib::RendererType rendererMode = renderlib::RendererType_Pathtrace,
                 QOpenGLContext* glContext = nullptr);

  void run();

  void wakeUp();

  void addRequest(RenderRequest* request);
  bool processRequest();

  inline int getTotalQueueDuration() { return this->m_totalQueueDuration; }

  inline int getRequestCount() { return this->m_requests.count(); }

  // 1 = continuous re-render, 0 = only wait for redraw commands
  virtual void setStreamMode(int32_t mode) { m_streamMode = mode > 0 ? true : false; }

  virtual void resizeGL(int internalWidth, int internalHeight);

protected:
  QString m_id;

  void processCommandBuffer(RenderRequest* rr);
  QImage render();

  void reset(int from = 0);

  int getTime();

  // this is a task queue
  QList<RenderRequest*> m_requests;
  QMutex m_requestMutex;
  QWaitCondition m_wait;

  int m_totalQueueDuration;

  void init();
  void shutDown();

private:
  QMutex* m_openGLMutex;

  RendererGLContext m_rglContext;

  GLFramebufferObject* m_fbo;

  std::atomic<bool> m_streamMode;
  int32_t m_width, m_height;

  QElapsedTimer m_time;

  struct myVolumeData
  {
    bool ownRenderer;
    RenderSettings* m_renderSettings;
    IRenderWindow* m_renderer;
    Scene* m_scene;
    CCamera* m_camera;
    LoadSpec m_loadSpec;
    Gesture m_gesture;
    GestureRendererGL m_gestureRenderer;

    myVolumeData()
      : m_camera(nullptr)
      , m_scene(nullptr)
      , m_renderSettings(nullptr)
      , m_renderer(nullptr)
      , ownRenderer(false)
    {
    }
  } m_myVolumeData;

  ExecutionContext m_ec;

signals:
  void requestProcessed(RenderRequest* request, QImage img);
  void sendString(RenderRequest* request, QString s);
};

#endif // RENDERER_H
