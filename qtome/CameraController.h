#pragma once


#include "Flags.h"
#include "Camera.h"

class RenderSettings;
class QMouseEvent;

// Define interaction style for controlling a realistic camera
class CameraController
{
public:
	CameraController(QCamera* cam);

	enum EMouseButtonFlag
	{
		Left	= 0x0001,
		Middle	= 0x0002,
		Right	= 0x0004
	};

	void setRenderSettings(RenderSettings& rs) { _renderSettings = &rs; }

	virtual void OnMouseWheelForward(void);
	virtual void OnMouseWheelBackward(void);
	virtual void OnMouseMove(QMouseEvent *event);

	int m_OldPos[2];
	int m_NewPos[2];

	// Camera sensitivity to mouse movement
	static float m_OrbitSpeed;			
	static float m_PanSpeed;
	static float m_ZoomSpeed;
	static float m_ContinuousZoomSpeed;
	static float m_ApertureSpeed;
	static float m_FovSpeed;

	RenderSettings* _renderSettings;
	QCamera* _camera;
};