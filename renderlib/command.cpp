#include "command.h"

#include "Scene.h"

void SessionCommand::execute(ExecutionContext* c) {

}
void AssetPathCommand::execute(ExecutionContext* c) {

}
void LoadOmeTifCommand::execute(ExecutionContext* c) {

}
void SetCameraPosCommand::execute(ExecutionContext* c) {

}
void SetCameraTargetCommand::execute(ExecutionContext* c) {

}
void SetCameraUpCommand::execute(ExecutionContext* c) {

}
void SetCameraApertureCommand::execute(ExecutionContext* c) {

}
void SetCameraFovYCommand::execute(ExecutionContext* c) {

}
void SetCameraFocalDistanceCommand::execute(ExecutionContext* c) {

}
void SetCameraExposureCommand::execute(ExecutionContext* c) {

}
void SetDiffuseColorCommand::execute(ExecutionContext* c) {
	c->_scene->m_DiffuseColor[0] = _data._r;
	c->_scene->m_DiffuseColor[1] = _data._g;
	c->_scene->m_DiffuseColor[2] = _data._b;
	c->_scene->m_DiffuseColor[3] = _data._a;
}
void SetSpecularColorCommand::execute(ExecutionContext* c) {
	c->_scene->m_SpecularColor[0] = _data._r;
	c->_scene->m_SpecularColor[1] = _data._g;
	c->_scene->m_SpecularColor[2] = _data._b;
	c->_scene->m_SpecularColor[3] = _data._a;
}
void SetEmissiveColorCommand::execute(ExecutionContext* c) {
	c->_scene->m_EmissiveColor[0] = _data._r;
	c->_scene->m_EmissiveColor[1] = _data._g;
	c->_scene->m_EmissiveColor[2] = _data._b;
	c->_scene->m_EmissiveColor[3] = _data._a;
}
void SetRenderIterationsCommand::execute(ExecutionContext* c) {

}
void SetStreamModeCommand::execute(ExecutionContext* c) {

}
void RequestRedrawCommand::execute(ExecutionContext* c) {

}
