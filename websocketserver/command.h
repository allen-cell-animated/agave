#pragma once

#include <string>

class CScene;
class Renderer;

struct ExecutionContext {
	Renderer* _renderer;
	CScene* _scene;
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
	float _r, _g, _b, _a;
};
CMDDECL(SetDiffuseColorCommand);

struct SetSpecularColorCommandD {
	float _r, _g, _b, _a;
};
CMDDECL(SetSpecularColorCommand);

struct SetEmissiveColorCommandD {
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

struct SetChannelCommandD {
	int32_t _x;
};
CMDDECL(SetChannelCommand);
