#pragma once

#include <string>

class CCamera;
class Renderer;
class RenderSettings;
class Scene;

struct ExecutionContext {
	Renderer* _renderer;
	RenderSettings* _renderSettings;
	Scene* _appScene;
	CCamera* _camera;
};

class Command {
public:
	// return number of bytes advanced
	virtual void execute(ExecutionContext* context) = 0;
};

#define CMDDECL(NAME)\
	class NAME : public Command {\
		public:\
			NAME(NAME##D d) : _data(d) {}\
			virtual void execute(ExecutionContext* context);\
			NAME##D _data;\
	};

struct SessionCommandD {
	std::string _name;
};
CMDDECL(SessionCommand);

struct AssetPathCommandD {
	std::string _name;
};
CMDDECL(AssetPathCommand);

struct LoadOmeTifCommandD {
	std::string _name;
};
CMDDECL(LoadOmeTifCommand);

struct SetCameraPosCommandD {
	float _x, _y, _z;
};
CMDDECL(SetCameraPosCommand);

struct SetCameraUpCommandD {
	float _x, _y, _z;
};
CMDDECL(SetCameraUpCommand);

struct SetCameraTargetCommandD {
	float _x, _y, _z;
};
CMDDECL(SetCameraTargetCommand);

struct SetCameraApertureCommandD {
	float _x;
};
CMDDECL(SetCameraApertureCommand);

struct SetCameraFovYCommandD {
	float _x;
};
CMDDECL(SetCameraFovYCommand);

struct SetCameraFocalDistanceCommandD {
	float _x;
};
CMDDECL(SetCameraFocalDistanceCommand);

struct SetCameraExposureCommandD {
	float _x;
};
CMDDECL(SetCameraExposureCommand);

struct SetDiffuseColorCommandD {
	int32_t _channel;
	float _r, _g, _b, _a;
};
CMDDECL(SetDiffuseColorCommand);

struct SetSpecularColorCommandD {
	int32_t _channel;
	float _r, _g, _b, _a;
};
CMDDECL(SetSpecularColorCommand);

struct SetEmissiveColorCommandD {
	int32_t _channel;
	float _r, _g, _b, _a;
};
CMDDECL(SetEmissiveColorCommand);

struct SetRenderIterationsCommandD {
	int32_t _x;
};
CMDDECL(SetRenderIterationsCommand);

struct SetStreamModeCommandD {
	int32_t _x;
};
CMDDECL(SetStreamModeCommand);

struct RequestRedrawCommandD {
};
CMDDECL(RequestRedrawCommand);

struct SetResolutionCommandD {
	int32_t _x, _y;
};
CMDDECL(SetResolutionCommand);

struct SetDensityCommandD {
	float _x;
};
CMDDECL(SetDensityCommand);

struct FrameSceneCommandD {
};
CMDDECL(FrameSceneCommand);

struct SetGlossinessCommandD {
	int32_t _channel;
	float _glossiness;
};
CMDDECL(SetGlossinessCommand);

struct EnableChannelCommandD {
	int32_t _channel;
	int32_t _enabled;
};
CMDDECL(EnableChannelCommand);

struct SetWindowLevelCommandD {
	int32_t _channel;
	float _window;
	float _level;
};
CMDDECL(SetWindowLevelCommand);

struct OrbitCameraCommandD {
	float _theta;
	float _phi;
};
CMDDECL(OrbitCameraCommand);
