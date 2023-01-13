#ifndef RENDERER_H
#define RENDERER_H

#include "glad/glad.h"

#include "renderlib/command.h"
#include "renderlib/gl/Util.h"
#include "renderlib/renderlib.h"
#include "renderrequest.h"

#include <QList>
#include <QMutex>
#include <QObject>
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QOpenGLTexture>
#include <QThread>
#include <QWaitCondition>
#include <QtCore/QElapsedTimer>

#include <memory>

class commandBuffer;
class CCamera;
class ImageXYZC;
class IRenderWindow;
class RenderSettings;
class Scene;

class RendererGLContext
{
public:
  RendererGLContext();
  ~RendererGLContext();
  void configure(QOpenGLContext* glContext = nullptr);
  void init();
  void destroy();

  void makeCurrent();
  void doneCurrent();

private:
  bool m_ownGLContext;
#if HAS_EGL
  HeadlessGLContext* m_glContext;
#else
  QOpenGLContext* m_glContext;
  QOffscreenSurface* m_surface;
#endif
};

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
                 std::string volumeFilePath = "",
                 int fileCurrentScene = 0,
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

  class SceneDescription
  {
  public:
    inline SceneDescription(QString name, int start, int end)
      : m_name(name)
      , m_start(start)
      , m_end(end)
    {
    }

    QString m_name;
    int m_start;
    int m_end;
  };

  QList<SceneDescription> m_scenes;

  // TODO move this info.  This class only knows about some abstract renderer and a scene object.
  void myVolumeInit();
  struct myVolumeData
  {
    bool ownRenderer;
    RenderSettings* m_renderSettings;
    IRenderWindow* m_renderer;
    Scene* m_scene;
    CCamera* m_camera;
    std::string mVolumeFilePath;

    myVolumeData()
      : m_camera(nullptr)
      , m_scene(nullptr)
      , m_renderSettings(nullptr)
      , m_renderer(nullptr)
      , ownRenderer(false)
      , mVolumeFilePath("")
    {
    }
  } m_myVolumeData;

  ExecutionContext m_ec;

signals:
  void kill();
  void requestProcessed(RenderRequest* request, QImage img);
  void sendString(RenderRequest* request, QString s);
};

#endif // RENDERER_H
