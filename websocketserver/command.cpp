#include "command.h"

#include "renderer.h"

#include "renderlib/AppScene.h"
#include "renderlib/FileReader.h"
#include "renderlib/ImageXYZC.h"
#include "renderlib/Logging.h"
#include "renderlib/Scene.h"

#include <QElapsedTimer>
#include <QFileInfo>

void SessionCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "Session command: " << _data._name;
}

void AssetPathCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "AssetPath command: " << _data._name;
}

void LoadOmeTifCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "LoadOmeTif command: " << _data._name;
	QFileInfo info(QString(_data._name.c_str()));
	if (info.exists())
	{
		FileReader fileReader;
		QElapsedTimer t;
		t.start();
		std::shared_ptr<ImageXYZC> image = fileReader.loadOMETiff_4D(_data._name);
		LOG_DEBUG << "Loaded " << _data._name << " in " << t.elapsed() << "ms";

		// pass this image to the renderer.
		c->_renderer->setImage(image);
		for (uint32_t i = 0; i < image->sizeC(); ++i) {
			c->_appScene->_material.enabled[i] = (i < 3);
		}
		c->_appScene->initSceneFromImg(image->sizeX(), image->sizeY(), image->sizeZ(),
			image->physicalSizeX(), image->physicalSizeY(), image->physicalSizeZ());
		c->_scene->initSceneFromImg(image->sizeX(), image->sizeY(), image->sizeZ(),
			image->physicalSizeX(), image->physicalSizeY(), image->physicalSizeZ());
		c->_scene->SetNoIterations(0);
	}
}

void SetCameraPosCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetCameraPos";
	c->_scene->m_Camera.m_From.x = _data._x;
	c->_scene->m_Camera.m_From.y = _data._y;
	c->_scene->m_Camera.m_From.z = _data._z;
	c->_scene->SetNoIterations(0);
}
void SetCameraTargetCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetCameraTarget";
	c->_scene->m_Camera.m_Target.x = _data._x;
	c->_scene->m_Camera.m_Target.y = _data._y;
	c->_scene->m_Camera.m_Target.z = _data._z;
	c->_scene->SetNoIterations(0);
}
void SetCameraUpCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetCameraUp";
	c->_scene->m_Camera.m_Up.x = _data._x;
	c->_scene->m_Camera.m_Up.y = _data._y;
	c->_scene->m_Camera.m_Up.z = _data._z;
	c->_scene->SetNoIterations(0);
}
void SetCameraApertureCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetCameraAperture";
	c->_scene->m_Camera.m_Aperture.m_Size = _data._x;
	c->_scene->SetNoIterations(0);
}
void SetCameraFovYCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetCameraFovY";
	c->_scene->m_Camera.m_FovV = _data._x;
	c->_scene->SetNoIterations(0);
}
void SetCameraFocalDistanceCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetCameraFocalDistance";
	c->_scene->m_Camera.m_Focus.m_FocalDistance = _data._x;
	c->_scene->SetNoIterations(0);
}
void SetCameraExposureCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetCameraExposure";
	// 0 is darkness, 1 is max
	c->_scene->m_Camera.m_Film.m_Exposure = 1.0f - _data._x;
	c->_scene->SetNoIterations(0);
}
void SetDiffuseColorCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetDiffuse";
	c->_appScene->_material.diffuse[_data._channel * 3 + 0] = _data._r;
	c->_appScene->_material.diffuse[_data._channel * 3 + 1] = _data._g;
	c->_appScene->_material.diffuse[_data._channel * 3 + 2] = _data._b;
	c->_scene->SetNoIterations(0);
}
void SetSpecularColorCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetSpecular";
	c->_appScene->_material.specular[_data._channel * 3 + 0] = _data._r;
	c->_appScene->_material.specular[_data._channel * 3 + 1] = _data._g;
	c->_appScene->_material.specular[_data._channel * 3 + 2] = _data._b;
	c->_scene->SetNoIterations(0);
}
void SetEmissiveColorCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetEmissive";
	c->_appScene->_material.emissive[_data._channel * 3 + 0] = _data._r;
	c->_appScene->_material.emissive[_data._channel * 3 + 1] = _data._g;
	c->_appScene->_material.emissive[_data._channel * 3 + 2] = _data._b;
	c->_scene->SetNoIterations(0);
}
void SetRenderIterationsCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetRenderIterations";
	c->_scene->m_Camera.m_Film.m_ExposureIterations = _data._x;
}
void SetStreamModeCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetStreamMode";
	c->_renderer->setStreamMode(_data._x);
}
void RequestRedrawCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "RequestRedraw";
	//	c->_renderer->renderNow();
}
void SetResolutionCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetResolution";
	c->_renderer->resizeGL(_data._x, _data._y);
}
void SetDensityCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetDensity";
	c->_scene->m_DensityScale = _data._x;
	c->_scene->SetNoIterations(0);
}

void FrameSceneCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "FrameScene";
	c->_scene->m_Camera.SetViewMode(ViewModeFront);
	c->_scene->SetNoIterations(0);
}
void SetGlossinessCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetGlossiness";
	c->_appScene->_material.roughness[_data._channel] = _data._glossiness;
	c->_scene->m_DirtyFlags.SetFlag(TransferFunctionDirty);
}
void EnableChannelCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "EnableChannel";
	// 0 or 1 hopefully.
	c->_appScene->_material.enabled[_data._channel] = (_data._enabled != 0);
	c->_scene->m_DirtyFlags.SetFlag(VolumeDataDirty);
}
void SetWindowLevelCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetWindowLevel";
	c->_appScene->_volume->channel(_data._channel)->generate_windowLevel(_data._window, _data._level);
	c->_scene->m_DirtyFlags.SetFlag(TransferFunctionDirty);
}
