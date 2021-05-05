#pragma once

#include "glad/glad.h"

#include "Logging.h"

#include <QImage>
#include <string>

/**
 * Check OpenGL status.
 *
 * Call glGetError() and log the specified message with
 * additional details if a problem was encountered.
 *
 * @param message the message to log on error.
 */
extern void
check_gl(std::string const& message);
extern bool
check_glfb(std::string const& message);

class GLImageShader2DnoLut;
class RectImage2D
{

public:
  RectImage2D();
  ~RectImage2D();

  void draw(GLuint texture2d);

private:
  /// The vertex array.
  GLuint _quadVertexArray; // vao
  /// The image vertices.
  GLuint _quadVertices; // buffer
  /// The image texture coordinates.
  GLuint _quadTexcoords; // buffer
  /// The image elements.
  GLuint _quadIndices; // buffer
  size_t _num_image_elements;

  GLImageShader2DnoLut* _image_shader;
};

class GLTimer
{
public:
  GLTimer(void);
  virtual ~GLTimer(void);

  void StartTimer(void);
  float StopTimer(void);
  float ElapsedTime(void);

private:
  bool m_Started;
  GLuint m_EventStart;
  GLuint m_EventStop;

  void synchronize(GLuint eventid);
  void eventElapsedTime(float* result, GLuint startEvent, GLuint stopEvent);
};

// RAII; must have a current gl context at creation time.
class GLFramebufferObject
{
public:
  GLFramebufferObject(int width, int height, GLint colorInternalFormat);

  ~GLFramebufferObject();

  void bind();
  void release();
  int width() const;
  int height() const;
  QImage toImage(bool include_alpha = false);

private:
  GLuint m_fbo;
  GLuint m_texture;
  GLuint m_depth_buffer;
  int m_width;
  int m_height;
};
