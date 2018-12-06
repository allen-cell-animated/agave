#pragma once

#include "glad/include/glad/glad.h"

#include <QtGui/QOpenGLDebugMessage>
#include <QtGui/QOpenGLFunctions_3_3_Core>
#include <QtGui/QWindow>

QT_BEGIN_NAMESPACE
class QPainter;
class QOpenGLContext;
class QOpenGLPaintDevice;
class QOpenGLDebugLogger;
QT_END_NAMESPACE

/**
 * Top level GL window.
 *
 * This is a standard QWindow, however it contains no child
 * widgets; it's simply a surface on which to paint GL rendered
 * content.
 */
class GLWindow : public QWindow
{
  Q_OBJECT

public:
  /**
   * Create a GL window.
   *
   * @param parent the parent of this object.
   */
  explicit GLWindow(QWindow* parent = 0);

  /// Destructor.
  ~GLWindow();

  /**
   * Render using a QPainter.
   *
   * Does nothing by default.  Subclass to paint with QPainter on
   * the OpenGL paint device.
   *
   * @param painter the painter for rendering.
   */
  virtual void render(QPainter* painter);

  /**
   * Render using OpenGL.
   *
   * By default sets up a QOpenGLPaintDevice and calls
   * render(QPainter*).  Subclass and reimplement to handle
   * rendering directly.
   */
  virtual void render();

  /**
   * Handle initialization of the window.
   */
  virtual void initialize();

  /**
   * Handle resizing of the window.
   */
  virtual void resize();

  /**
   * Enable or disable animating.
   *
   * If enabled, this will trigger a full render pass for every
   * frame using a timer.  If disabled, rendering must be
   * triggered by hand.
   *
   * @param animating @c true to enable continuous animation or @c
   * false to disable.
   */
  void setAnimating(bool animating);

public slots:
  /**
   * Render a frame at the next opportunity.
   *
   * Mark the window for requiring a full render pass at a future
   * point in time.  This will usually be for the next frame.
   */
  void renderLater();

  /**
   * Render a frame immediately.
   *
   * This method also handles initialization of the GL context if
   * it has not been called previously.
   */
  void renderNow();

  /**
   * Log a GL debug message.
   *
   * This currently logs to stderr due to the high log volume when
   * debugging is enabled.
   *
   * @param message the message to log.
   */
  void logMessage(QOpenGLDebugMessage message);

protected:
  /**
   * Handle events.
   *
   * Used to handle timer events and trigger a rendering pass.
   *
   * @param event the event to handle.
   * @returns @c true if the event was handled.
   */
  bool event(QEvent* event);

  /**
   * Handle expose events.
   *
   * Trigger a rendering pass on exposure.
   *
   * @param event the event to handle.
   */
  void exposeEvent(QExposeEvent* event);

  /**
   * Handle resize events.
   *
   * @param event the event to handle.
   */
  void resizeEvent(QResizeEvent* event);

  /**
   * Get GL context.
   *
   * @returns the GL context.
   */
  QOpenGLContext* context() const;

  /**
   * Make the GL context for this window the current context.
   */
  void makeCurrent();

private:
  /// Update at next opportunity?
  bool m_update_pending;
  /// Animation enabled?
  bool m_animating;
  /// OpenGL context.
  QOpenGLContext* m_glcontext;
  /// OpenGL paint device (if render is not reimplemented in subclass).
  QOpenGLPaintDevice* m_device;
  /// OpenGL debug logger (if logging enabled).
  QOpenGLDebugLogger* m_logger;
};
