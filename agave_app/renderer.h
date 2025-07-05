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
#include <QStandardPaths>
#include <QThread>
#include <QWaitCondition>

#include <memory>

class commandBuffer;
class CCamera;
class ImageXYZC;
class IRenderWindow;
class RenderSettings;
class Scene;

// serialized so permanent?
enum eRenderDurationType
{
  TIME = 0,
  SAMPLES = 1
};

struct RenderDuration
{
  int samples = 1;
  int duration = 0; // in seconds
  eRenderDurationType durationType = SAMPLES;
};

struct CaptureSettings
{
  std::string outputDir;
  std::string filenamePrefix;
  int width;
  int height;
  RenderDuration renderDuration;
  int startTime;
  int endTime;

  CaptureSettings()
  {
    // defaults!
    QString docs = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
    outputDir = docs.toStdString();

    filenamePrefix = "frame";
    width = 0;
    height = 0;
    renderDuration.duration = 10;
    renderDuration.samples = 32;
    renderDuration.durationType = SAMPLES;
    startTime = 0;
    endTime = 0;
  }

  CaptureSettings(const CaptureSettings& other)
  {
    outputDir = other.outputDir;
    filenamePrefix = other.filenamePrefix;
    width = other.width;
    height = other.height;
    renderDuration = other.renderDuration;
    startTime = other.startTime;
    endTime = other.endTime;
  }
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
                 std::shared_ptr<RenderSettings> renderSettings,
                 const Scene& scene,
                 const CCamera& camera,
                 const LoadSpec& loadSpec,
                 // rendererMode ignored if renderer is non-null
                 renderlib::RendererType rendererMode = renderlib::RendererType_Pathtrace,
                 QOpenGLContext* glContext = nullptr,
                 const CaptureSettings* captureSettings = nullptr);

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

  int32_t m_frameIterations;
  float m_frameTimeSeconds;
  bool shouldContinue();

  int32_t m_width, m_height;

  QElapsedTimer m_time;

  struct myVolumeData
  {
    bool ownRenderer;
    RenderSettings* m_renderSettings;
    CaptureSettings* m_captureSettings;
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
      , m_captureSettings(nullptr)
      , m_renderer(nullptr)
      , ownRenderer(false)
    {
    }
  } m_myVolumeData;

  ExecutionContext m_ec;

signals:
  void requestProcessed(RenderRequest* request, QImage img);
  void frameDone(QImage img);
  void sendString(RenderRequest* request, QString s);
};

#endif // RENDERER_H
