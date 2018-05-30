#include "command.h"

#include "renderer.h"

#include "renderlib/AppScene.h"
#include "renderlib/CCamera.h"
#include "renderlib/FileReader.h"
#include "renderlib/ImageXYZC.h"
#include "renderlib/Logging.h"
#include "renderlib/RenderSettings.h"

#include <QElapsedTimer>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

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

		c->_appScene->_volume = image;
		c->_appScene->initSceneFromImg(image);

		// Tell the camera about the volume's bounding box
		c->_camera->m_SceneBoundingBox.m_MinP = c->_appScene->_boundingBox.GetMinP();
		c->_camera->m_SceneBoundingBox.m_MaxP = c->_appScene->_boundingBox.GetMaxP();
		c->_camera->SetViewMode(ViewModeFront);

		for (uint32_t i = 0; i < image->sizeC(); ++i) {
			c->_appScene->_material.enabled[i] = (i < 3);
		}
		c->_renderSettings->SetNoIterations(0);
		c->_renderSettings->m_DirtyFlags.SetFlag(VolumeDirty);
		c->_renderSettings->m_DirtyFlags.SetFlag(VolumeDataDirty);

		// fire back some json immediately...
		QJsonObject j;
		j["commandId"] = (int)LoadOmeTifCommand::_ID;
		j["x"] = (int)image->sizeX();
		j["y"] = (int)image->sizeY();
		j["z"] = (int)image->sizeZ();
		j["c"] = (int)image->sizeC();
		j["t"] = 1;
		j["pixel_size_x"] = image->physicalSizeX();
		j["pixel_size_y"] = image->physicalSizeY();
		j["pixel_size_z"] = image->physicalSizeZ();
		QJsonArray channelNames;
		for (uint32_t i = 0; i < image->sizeC(); ++i) {
			channelNames.append(image->channel(i)->_name);
		}
		j["channel_names"] = channelNames;
			
		QJsonDocument doc(j);
		c->_message = doc.toJson();
	}
}

void SetCameraPosCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetCameraPos " << _data._x << " " << _data._y << " " << _data._z;
	c->_camera->m_From.x = _data._x;
	c->_camera->m_From.y = _data._y;
	c->_camera->m_From.z = _data._z;
	c->_renderSettings->SetNoIterations(0);
}
void SetCameraTargetCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetCameraTarget " << _data._x << " " << _data._y << " " << _data._z;
	c->_camera->m_Target.x = _data._x;
	c->_camera->m_Target.y = _data._y;
	c->_camera->m_Target.z = _data._z;
	c->_renderSettings->SetNoIterations(0);
}
void SetCameraUpCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetCameraUp " << _data._x << " " << _data._y << " " << _data._z;
	c->_camera->m_Up.x = _data._x;
	c->_camera->m_Up.y = _data._y;
	c->_camera->m_Up.z = _data._z;
	c->_renderSettings->SetNoIterations(0);
}
void SetCameraApertureCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetCameraAperture " << _data._x;
	c->_camera->m_Aperture.m_Size = _data._x;
	c->_renderSettings->SetNoIterations(0);
}
void SetCameraFovYCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetCameraFovY " << _data._x;
	c->_camera->m_FovV = _data._x;
	c->_renderSettings->SetNoIterations(0);
}
void SetCameraFocalDistanceCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetCameraFocalDistance " << _data._x;

	// TODO: how will we ever set the camera back to auto focus?
	c->_camera->m_Focus.m_Type = CFocus::Manual;
	
	c->_camera->m_Focus.m_FocalDistance = _data._x;
	c->_renderSettings->SetNoIterations(0);
}
void SetCameraExposureCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetCameraExposure " << _data._x;
	// 0 is darkness, 1 is max
	c->_camera->m_Film.m_Exposure = 1.0f - _data._x;
	c->_renderSettings->SetNoIterations(0);
}
void SetDiffuseColorCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetDiffuse " << _data._channel << " " << _data._r << " " << _data._g << " " << _data._b;
	c->_appScene->_material.diffuse[_data._channel * 3 + 0] = _data._r;
	c->_appScene->_material.diffuse[_data._channel * 3 + 1] = _data._g;
	c->_appScene->_material.diffuse[_data._channel * 3 + 2] = _data._b;
	c->_renderSettings->SetNoIterations(0);
}
void SetSpecularColorCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetSpecular " << _data._channel << " " << _data._r << " " << _data._g << " " << _data._b;
	c->_appScene->_material.specular[_data._channel * 3 + 0] = _data._r;
	c->_appScene->_material.specular[_data._channel * 3 + 1] = _data._g;
	c->_appScene->_material.specular[_data._channel * 3 + 2] = _data._b;
	c->_renderSettings->SetNoIterations(0);
}
void SetEmissiveColorCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetEmissive " << _data._channel << " " << _data._r << " " << _data._g << " " << _data._b;
	c->_appScene->_material.emissive[_data._channel * 3 + 0] = _data._r;
	c->_appScene->_material.emissive[_data._channel * 3 + 1] = _data._g;
	c->_appScene->_material.emissive[_data._channel * 3 + 2] = _data._b;
	c->_renderSettings->SetNoIterations(0);
}
void SetRenderIterationsCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetRenderIterations " << _data._x;
	c->_camera->m_Film.m_ExposureIterations = _data._x;
}
void SetStreamModeCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetStreamMode " << _data._x;
	c->_renderer->setStreamMode(_data._x);
}
void RequestRedrawCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "RequestRedraw";
	//	c->_renderer->renderNow();
}
void SetResolutionCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetResolution " << _data._x << " " << _data._y;
	c->_camera->m_Film.m_Resolution.SetResX(_data._x);
	c->_camera->m_Film.m_Resolution.SetResY(_data._y);
	c->_renderer->resizeGL(_data._x, _data._y);
	c->_renderSettings->SetNoIterations(0);
}
void SetDensityCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetDensity " << _data._x;
	c->_renderSettings->m_RenderSettings.m_DensityScale = _data._x;
	c->_renderSettings->SetNoIterations(0);
}

void FrameSceneCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "FrameScene";
	c->_camera->m_SceneBoundingBox.m_MinP = c->_appScene->_boundingBox.GetMinP();
	c->_camera->m_SceneBoundingBox.m_MaxP = c->_appScene->_boundingBox.GetMaxP();
	c->_camera->SetViewMode(ViewModeFront);
	c->_renderSettings->SetNoIterations(0);
}
void SetGlossinessCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetGlossiness " << _data._channel << " " << _data._glossiness;
	c->_appScene->_material.roughness[_data._channel] = _data._glossiness;
	c->_renderSettings->m_DirtyFlags.SetFlag(TransferFunctionDirty);
}
void EnableChannelCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "EnableChannel " << _data._channel << " " << _data._enabled;
	// 0 or 1 hopefully.
	c->_appScene->_material.enabled[_data._channel] = (_data._enabled != 0);
	c->_renderSettings->m_DirtyFlags.SetFlag(VolumeDataDirty);
}
void SetWindowLevelCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetWindowLevel " << _data._channel << " " << _data._window << " " << _data._level;
	c->_appScene->_volume->channel(_data._channel)->generate_windowLevel(_data._window, _data._level);
	c->_renderSettings->m_DirtyFlags.SetFlag(TransferFunctionDirty);
}
void OrbitCameraCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "OrbitCamera " << _data._theta << " " << _data._phi;
	c->_camera->Orbit(_data._theta, _data._phi);
	c->_renderSettings->SetNoIterations(0);
}
void SetSkylightTopColorCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetSkylightTopColor " << _data._r << " " << _data._g << " " << _data._b;
	c->_appScene->_lighting.m_Lights[0].m_ColorTop = glm::vec3(_data._r, _data._g, _data._b);
	c->_renderSettings->m_DirtyFlags.SetFlag(LightsDirty);
}
void SetSkylightMiddleColorCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetSkylightMiddleColor " << _data._r << " " << _data._g << " " << _data._b;
	c->_appScene->_lighting.m_Lights[0].m_ColorMiddle = glm::vec3(_data._r, _data._g, _data._b);
	c->_renderSettings->m_DirtyFlags.SetFlag(LightsDirty);
}
void SetSkylightBottomColorCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetSkylightBottomColor " << _data._r << " " << _data._g << " " << _data._b;
	c->_appScene->_lighting.m_Lights[0].m_ColorBottom = glm::vec3(_data._r, _data._g, _data._b);
	c->_renderSettings->m_DirtyFlags.SetFlag(LightsDirty);
}
void SetLightPosCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetLightPos " << _data._r << " " << _data._theta << " " << _data._phi;
	c->_appScene->_lighting.m_Lights[1 + _data._index].m_Distance = _data._r;
	c->_appScene->_lighting.m_Lights[1 + _data._index].m_Theta = _data._theta;
	c->_appScene->_lighting.m_Lights[1 + _data._index].m_Phi = _data._phi;
	c->_renderSettings->m_DirtyFlags.SetFlag(LightsDirty);
}
void SetLightColorCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetLightColor " << _data._index << " " << _data._r << " " << _data._g << " " << _data._b;
	c->_appScene->_lighting.m_Lights[1 + _data._index].m_Color = glm::vec3(_data._r, _data._g, _data._b);
	c->_renderSettings->m_DirtyFlags.SetFlag(LightsDirty);
}
void SetLightSizeCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetLightSize " << _data._index << " " << _data._x << " " << _data._y;
	c->_appScene->_lighting.m_Lights[1 + _data._index].m_Width = _data._x;
	c->_appScene->_lighting.m_Lights[1 + _data._index].m_Height = _data._y;
	c->_renderSettings->m_DirtyFlags.SetFlag(LightsDirty);
}
void SetClipRegionCommand::execute(ExecutionContext* c) {
	LOG_DEBUG << "SetClipRegion " << _data._minx << " " << _data._maxx << " " << _data._miny << " " << _data._maxy << " " << _data._minz << " " << _data._maxz;
	c->_appScene->_roi.SetMinP(glm::vec3(_data._minx, _data._miny, _data._minz));
	c->_appScene->_roi.SetMaxP(glm::vec3(_data._maxx, _data._maxy, _data._maxz));
	c->_renderSettings->m_DirtyFlags.SetFlag(RoiDirty);
}

