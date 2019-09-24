#include "CameraController.h"

#include "RenderSettings.h"

// renderlib
#include "renderlib/CCamera.h"
#include "renderlib/Logging.h"

#include <QApplication>
#include <QDesktopWidget>
#include <QtGui/QGuiApplication>
#include <QtGui/QMouseEvent>

float CameraController::m_OrbitSpeed = 1.0f;
float CameraController::m_PanSpeed = 1.0f;
float CameraController::m_ZoomSpeed = 1000.0f;
float CameraController::m_ContinuousZoomSpeed = 0.0000001f;
float CameraController::m_ApertureSpeed = 0.001f;
float CameraController::m_FovSpeed = 0.5f;

CameraController::CameraController(QCamera* cam, CCamera* theCamera)
  : m_renderSettings(nullptr)
  , m_qcamera(cam)
  , m_CCamera(theCamera)
{}

void
CameraController::OnMouseWheelForward(void)
{
  m_CCamera->Zoom(-m_ZoomSpeed);

  // Flag the camera as dirty, this will restart the rendering
  m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
};

void
CameraController::OnMouseWheelBackward(void)
{
  m_CCamera->Zoom(m_ZoomSpeed);

  // Flag the camera as dirty, this will restart the rendering
  m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
};

bool
GetShiftKey()
{
  return QGuiApplication::keyboardModifiers() & Qt::ShiftModifier;
}
bool
GetCtrlKey()
{
  return QGuiApplication::keyboardModifiers() & Qt::ControlModifier;
}
bool
GetAltKey()
{
  return QGuiApplication::keyboardModifiers() & Qt::AltModifier;
}

void
CameraController::OnMouseMove(QMouseEvent* event)
{
  float devicePixelRatio = QApplication::desktop()->devicePixelRatioF();
  if (event->buttons() & Qt::LeftButton) {
    if (GetCtrlKey()) {
      // Zooming (Dolly): ctrl + left drag
      m_NewPos[0] = event->x();
      m_NewPos[1] = event->y();

      m_CCamera->Zoom(-(float)(m_NewPos[1] - m_OldPos[1]) / devicePixelRatio);

      m_OldPos[0] = event->x();
      m_OldPos[1] = event->y();

      // Flag the camera as dirty, this will restart the rendering
      m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
    } else if (GetAltKey()) {
      // Panning (tracking): alt + left drag
      m_NewPos[0] = event->x();
      m_NewPos[1] = event->y();

      m_CCamera->Pan(m_PanSpeed * (float)(m_NewPos[1] - m_OldPos[1]) / devicePixelRatio,
                     -m_PanSpeed * ((float)(m_NewPos[0] - m_OldPos[0]) / devicePixelRatio));

      m_OldPos[0] = event->x();
      m_OldPos[1] = event->y();

      // Flag the camera as dirty, this will restart the rendering
      m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
    } else {
      // Orbiting: unmodified left button

      m_NewPos[0] = event->x();
      m_NewPos[1] = event->y();

      if (m_NewPos[1] != m_OldPos[1] || m_NewPos[0] != m_OldPos[0]) {
        m_CCamera->Trackball(-m_OrbitSpeed * (float)(m_NewPos[1] - m_OldPos[1]) / devicePixelRatio,
                             -m_OrbitSpeed * (float)(m_NewPos[0] - m_OldPos[0]) / devicePixelRatio);

        m_OldPos[0] = event->x();
        m_OldPos[1] = event->y();

        // Flag the camera as dirty, this will restart the rendering
        m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
      }
    }
  }

  // Panning (tracking): middle mouse drag
  if (event->buttons() & Qt::MiddleButton) {
    m_NewPos[0] = event->x();
    m_NewPos[1] = event->y();

    m_CCamera->Pan(m_PanSpeed * (float)(m_NewPos[1] - m_OldPos[1]) / devicePixelRatio,
                   -m_PanSpeed * ((float)(m_NewPos[0] - m_OldPos[0]) / devicePixelRatio));

    m_OldPos[0] = event->x();
    m_OldPos[1] = event->y();

    // Flag the camera as dirty, this will restart the rendering
    m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
  }

  // Zooming (Dolly): right mouse drag
  if (event->buttons() & Qt::RightButton) {
    m_NewPos[0] = event->x();
    m_NewPos[1] = event->y();

    m_CCamera->Zoom(-(float)(m_NewPos[1] - m_OldPos[1]) / devicePixelRatio);

    m_OldPos[0] = event->x();
    m_OldPos[1] = event->y();

    // Flag the camera as dirty, this will restart the rendering
    m_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
  }
}
