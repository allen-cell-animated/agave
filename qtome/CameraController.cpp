#include "CameraController.h"

#include "RenderSettings.h"

// renderlib
#include "renderlib/Logging.h"
#include "renderlib/CCamera.h"

#include <QtGui/QGuiApplication>
#include <QtGui/QMouseEvent>

float CameraController::m_OrbitSpeed			= 1.0f;
float CameraController::m_PanSpeed				= 1.0f;
float CameraController::m_ZoomSpeed				= 1000.0f;
float CameraController::m_ContinuousZoomSpeed	= 0.0000001f;
float CameraController::m_ApertureSpeed			= 0.001f;
float CameraController::m_FovSpeed				= 0.5f;

CameraController::CameraController(QCamera* cam, CCamera* theCamera)
:	_renderSettings(nullptr),
	_camera(cam),
	mCamera(theCamera)
{

}

void CameraController::OnMouseWheelForward(void)
{
	mCamera->Zoom(-m_ZoomSpeed);

	// Flag the camera as dirty, this will restart the rendering
	_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
};
	
void CameraController::OnMouseWheelBackward(void)
{
	mCamera->Zoom(m_ZoomSpeed);

	// Flag the camera as dirty, this will restart the rendering
	_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
};

bool GetShiftKey() {
    return QGuiApplication::keyboardModifiers() & Qt::ShiftModifier;
}
bool GetCtrlKey() {
    return QGuiApplication::keyboardModifiers() & Qt::ControlModifier;
}

void CameraController::OnMouseMove(QMouseEvent *event) 
{
	// Orbiting
	if (event->buttons() & Qt::LeftButton)
	{
		if (GetShiftKey() && GetCtrlKey())
		{
            m_NewPos[0] = event->x();
            m_NewPos[1] = event->y();

			_camera->GetFocus().SetFocalDistance(max(0.0f, mCamera->m_Focus.m_FocalDistance + m_ApertureSpeed * (float)(m_NewPos[1] - m_OldPos[1])));

            m_OldPos[0] = event->x();
            m_OldPos[1] = event->y();

			// Flag the camera as dirty, this will restart the rendering
			_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
		}
		else
		{
			if (GetShiftKey())
			{
                m_NewPos[0] = event->x();
                m_NewPos[1] = event->y();

				_camera->GetAperture().SetSize(max(0.0f, mCamera->m_Aperture.m_Size + m_ApertureSpeed * (float)(m_NewPos[1] - m_OldPos[1])));

                m_OldPos[0] = event->x();
                m_OldPos[1] = event->y();

				// Flag the camera as dirty, this will restart the rendering
				_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
			}
			else if (GetCtrlKey())
			{
                m_NewPos[0] = event->x();
                m_NewPos[1] = event->y();

				_camera->GetProjection().SetFieldOfView(max(0.0f, mCamera->m_FovV - m_FovSpeed * (float)(m_NewPos[1] - m_OldPos[1])));

                m_OldPos[0] = event->x();
                m_OldPos[1] = event->y();

				/// Flag the camera as dirty, this will restart the rendering
				_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
			}
			else
			{
                m_NewPos[0] = event->x();
                m_NewPos[1] = event->y();

				mCamera->Orbit(-0.6f * m_OrbitSpeed * (float)(m_NewPos[1] - m_OldPos[1]), -m_OrbitSpeed * (float)(m_NewPos[0] - m_OldPos[0]));
				//LOG_TRACE << "Orbit Tgt " << _Scene->m_Camera.m_Target.x << " " << _Scene->m_Camera.m_Target.y << " " << _Scene->m_Camera.m_Target.z;
				//LOG_TRACE << "Orbit From " << _Scene->m_Camera.m_From.x << " " << _Scene->m_Camera.m_From.y << " " << _Scene->m_Camera.m_From.z;

                m_OldPos[0] = event->x();
                m_OldPos[1] = event->y();

				// Flag the camera as dirty, this will restart the rendering
				_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
			}
		}
	}

	// Panning
	if (event->buttons() & Qt::MiddleButton)
	{
        m_NewPos[0] = event->x();
        m_NewPos[1] = event->y();

		mCamera->Pan(m_PanSpeed * (float)(m_NewPos[1] - m_OldPos[1]), -m_PanSpeed * ((float)(m_NewPos[0] - m_OldPos[0])));

        m_OldPos[0] = event->x();
        m_OldPos[1] = event->y();

		// Flag the camera as dirty, this will restart the rendering
		_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
	}

	// Zooming
	if (event->buttons() & Qt::RightButton)
	{
        m_NewPos[0] = event->x();
        m_NewPos[1] = event->y();

		mCamera->Zoom(-(float)(m_NewPos[1] - m_OldPos[1]));

        m_OldPos[0] = event->x();
        m_OldPos[1] = event->y();

		// Flag the camera as dirty, this will restart the rendering
		_renderSettings->m_DirtyFlags.SetFlag(CameraDirty);
	}
}
