#include "command.h"

#include "renderer.h"

#include "renderlib/Logging.h"
#include "renderlib/Scene.h"

void SessionCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "Session command: " << _data._name;
}
void AssetPathCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "AssetPath command: " << _data._name;
}
void LoadOmeTifCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "LoadOmeTif command: " << _data._name;
}
void SetCameraPosCommand::execute(ExecutionContext* c) {
	c->_scene->m_Camera.m_From.x = _data._x;
	c->_scene->m_Camera.m_From.y = _data._y;
	c->_scene->m_Camera.m_From.z = _data._z;
	c->_scene->SetNoIterations(0);
}
void SetCameraTargetCommand::execute(ExecutionContext* c) {
	c->_scene->m_Camera.m_Target.x = _data._x;
	c->_scene->m_Camera.m_Target.y = _data._y;
	c->_scene->m_Camera.m_Target.z = _data._z;
	c->_scene->SetNoIterations(0);
}
void SetCameraUpCommand::execute(ExecutionContext* c) {
	c->_scene->m_Camera.m_Up.x = _data._x;
	c->_scene->m_Camera.m_Up.y = _data._y;
	c->_scene->m_Camera.m_Up.z = _data._z;
	c->_scene->SetNoIterations(0);
}
void SetCameraApertureCommand::execute(ExecutionContext* c) {
	c->_scene->m_Camera.m_Aperture.m_Size = _data._x;
	c->_scene->SetNoIterations(0);
}
void SetCameraFovYCommand::execute(ExecutionContext* c) {
	c->_scene->m_Camera.m_FovV = _data._x;
	c->_scene->SetNoIterations(0);
}
void SetCameraFocalDistanceCommand::execute(ExecutionContext* c) {
	c->_scene->m_Camera.m_Focus.m_FocalDistance = _data._x;
	c->_scene->SetNoIterations(0);
}
void SetCameraExposureCommand::execute(ExecutionContext* c) {
	// 0 is darkness, 1 is max
	c->_scene->m_Camera.m_Film.m_Exposure = 1.0f - _data._x;
	c->_scene->SetNoIterations(0);
}
void SetDiffuseColorCommand::execute(ExecutionContext* c) {
	c->_scene->m_DiffuseColor[0] = _data._r;
	c->_scene->m_DiffuseColor[1] = _data._g;
	c->_scene->m_DiffuseColor[2] = _data._b;
	c->_scene->m_DiffuseColor[3] = _data._a;
	c->_scene->SetNoIterations(0);
}
void SetSpecularColorCommand::execute(ExecutionContext* c) {
	c->_scene->m_SpecularColor[0] = _data._r;
	c->_scene->m_SpecularColor[1] = _data._g;
	c->_scene->m_SpecularColor[2] = _data._b;
	c->_scene->m_SpecularColor[3] = _data._a;
	c->_scene->SetNoIterations(0);
}
void SetEmissiveColorCommand::execute(ExecutionContext* c) {
	c->_scene->m_EmissiveColor[0] = _data._r;
	c->_scene->m_EmissiveColor[1] = _data._g;
	c->_scene->m_EmissiveColor[2] = _data._b;
	c->_scene->m_EmissiveColor[3] = _data._a;
	c->_scene->SetNoIterations(0);
}
void SetRenderIterationsCommand::execute(ExecutionContext* c) {
	c->_scene->m_Camera.m_Film.m_ExposureIterations = _data._x;
}
void SetStreamModeCommand::execute(ExecutionContext* c) {
//	c->_renderer->setStreamMode(_data._x);
}
void RequestRedrawCommand::execute(ExecutionContext* c) {
//	c->_renderer->renderNow();
}
void SetResolutionCommand::execute(ExecutionContext* c) {
	c->_renderer->resizeGL(_data._x, _data._y);
}
void SetChannelCommand::execute(ExecutionContext* c) {
	c->_scene->_channel = _data._x;
	c->_scene->SetNoIterations(0);
}
void SetDensityCommand::execute(ExecutionContext* c) {
	c->_scene->m_DensityScale = _data._x;
	c->_scene->SetNoIterations(0);
}
