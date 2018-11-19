#pragma once

#include <glm.h>

#include <QJsonDocument>
#include <QString>

#include <vector>

struct ChannelViewerState {
	bool _enabled;
	float _window, _level;
	float _opacity;
	float _glossiness;
	glm::vec3 _diffuse, _specular, _emissive;

	ChannelViewerState() : 
		_enabled(true),
		_window(1.0),
		_level(0.5),
		_opacity(1.0),
		_glossiness(0.0),
		_diffuse(0.5, 0.5, 0.5)
	{}
};

struct LightViewerState {
    int _type;
	float _theta, _phi;
	glm::vec3 _color;
	glm::vec3 _topColor, _middleColor, _bottomColor;
	float _width, _height, _distance;

	LightViewerState() :
		_theta(0), _phi(0),
		_color(0.5, 0.5, 0.5),
		_width(1), _height(1), _distance(10)
	{}
};

struct ViewerState {
	QString _volumeImageFile;
	std::vector<ChannelViewerState> _channels;
    int _resolutionX, _resolutionY;
    int _renderIterations;
	float _exposure;
	float _densityScale;
	float _fov;
	float _apertureSize;
    float _focalDistance;
	float _gradientFactor;
	float _primaryStepSize, _secondaryStepSize;
	float _roiXmax, _roiYmax, _roiZmax, _roiXmin, _roiYmin, _roiZmin;
	float _scaleX, _scaleY, _scaleZ;

    float _eyeX, _eyeY, _eyeZ;
    float _targetX, _targetY, _targetZ;
    float _upX, _upY, _upZ;

	LightViewerState _light0;
	LightViewerState _light1;

	QJsonDocument stateToJson() const;
	void stateFromJson(QJsonDocument& jsonDoc);

	static ViewerState readStateFromJson(QString filePath);
    static void writeStateToJson(QString filePath, const ViewerState& state);
};
