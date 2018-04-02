#pragma once

#include <string>
#include <QString>

class CCamera;
class Renderer;
class RenderSettings;
class Scene;

struct ExecutionContext {
	Renderer* _renderer;
	RenderSettings* _renderSettings;
	Scene* _appScene;
	CCamera* _camera;
	QString _message;
};

class Command {
public:
	// return number of bytes advanced
	virtual void execute(ExecutionContext* context) = 0;
};

#define CMDDECL(NAME, CMDID)\
	class NAME : public Command {\
		public:\
			NAME(NAME##D d) : _data(d) {}\
			virtual void execute(ExecutionContext* context);\
			static const uint32_t _ID = CMDID;\
			NAME##D _data;\
	};

struct SessionCommandD {
	std::string _name;
};
CMDDECL(SessionCommand, 0);

struct AssetPathCommandD {
	std::string _name;
};
CMDDECL(AssetPathCommand, 1);

struct LoadOmeTifCommandD {
	std::string _name;
};
CMDDECL(LoadOmeTifCommand, 2);

struct SetCameraPosCommandD {
	float _x, _y, _z;
};
CMDDECL(SetCameraPosCommand, 3);

struct SetCameraTargetCommandD {
	float _x, _y, _z;
};
CMDDECL(SetCameraTargetCommand, 4);

struct SetCameraUpCommandD {
	float _x, _y, _z;
};
CMDDECL(SetCameraUpCommand, 5);

struct SetCameraApertureCommandD {
	float _x;
};
CMDDECL(SetCameraApertureCommand, 6);

struct SetCameraFovYCommandD {
	float _x;
};
CMDDECL(SetCameraFovYCommand, 7);

struct SetCameraFocalDistanceCommandD {
	float _x;
};
CMDDECL(SetCameraFocalDistanceCommand, 8);

struct SetCameraExposureCommandD {
	float _x;
};
CMDDECL(SetCameraExposureCommand, 9);

struct SetDiffuseColorCommandD {
	int32_t _channel;
	float _r, _g, _b, _a;
};
CMDDECL(SetDiffuseColorCommand, 10);

struct SetSpecularColorCommandD {
	int32_t _channel;
	float _r, _g, _b, _a;
};
CMDDECL(SetSpecularColorCommand, 11);

struct SetEmissiveColorCommandD {
	int32_t _channel;
	float _r, _g, _b, _a;
};
CMDDECL(SetEmissiveColorCommand, 12);

struct SetRenderIterationsCommandD {
	int32_t _x;
};
CMDDECL(SetRenderIterationsCommand, 13);

struct SetStreamModeCommandD {
	int32_t _x;
};
CMDDECL(SetStreamModeCommand, 14);

struct RequestRedrawCommandD {
};
CMDDECL(RequestRedrawCommand, 15);

struct SetResolutionCommandD {
	int32_t _x, _y;
};
CMDDECL(SetResolutionCommand, 16);

struct SetDensityCommandD {
	float _x;
};
CMDDECL(SetDensityCommand, 17);

struct FrameSceneCommandD {
};
CMDDECL(FrameSceneCommand, 18);

struct SetGlossinessCommandD {
	int32_t _channel;
	float _glossiness;
};
CMDDECL(SetGlossinessCommand, 19);

struct EnableChannelCommandD {
	int32_t _channel;
	int32_t _enabled;
};
CMDDECL(EnableChannelCommand, 20);

struct SetWindowLevelCommandD {
	int32_t _channel;
	float _window;
	float _level;
};
CMDDECL(SetWindowLevelCommand, 21);

struct OrbitCameraCommandD {
	float _theta;
	float _phi;
};
CMDDECL(OrbitCameraCommand, 22);

struct SetSkylightTopColorCommandD {
	float _r, _g, _b;
};
CMDDECL(SetSkylightTopColorCommand, 23);

struct SetSkylightMiddleColorCommandD {
	float _r, _g, _b;
};
CMDDECL(SetSkylightMiddleColorCommand, 24);

struct SetSkylightBottomColorCommandD {
	float _r, _g, _b;
};
CMDDECL(SetSkylightBottomColorCommand, 25);

struct SetLightPosCommandD {
	int32_t _index;
	float _r, _theta, _phi;
};
CMDDECL(SetLightPosCommand, 26);

struct SetLightColorCommandD {
	int32_t _index;
	float _r, _g, _b;
};
CMDDECL(SetLightColorCommand, 27);

struct SetLightSizeCommandD {
	int32_t _index;
	float _x, _y;
};
CMDDECL(SetLightSizeCommand, 28);

struct SetClipRegionCommandD {
	float _minx, _maxx;
	float _miny, _maxy;
	float _minz, _maxz;
};
CMDDECL(SetClipRegionCommand, 29);


