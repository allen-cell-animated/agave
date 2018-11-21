#pragma once

#include <glm.h>

#include <QJsonDocument>
#include <QString>

#include <vector>

struct ChannelViewerState {
	bool _enabled = true;
	float _window = 1.0f, _level = 0.5f;
	float _opacity = 1.0f;
	float _glossiness = 0.0f;
	glm::vec3 _diffuse = glm::vec3(0.5f, 0.5f, 0.5f), _specular, _emissive;
};

struct LightViewerState {
    int _type = 0;
	float _theta = 0.0f, _phi = 0.0f;
	float _colorIntensity = 1.0;
	glm::vec3 _color = glm::vec3(0.5f, 0.5f, 0.5f);
	glm::vec3 _topColor = glm::vec3(0.5f, 0.5f, 0.5f),
		_middleColor = glm::vec3(0.5f, 0.5f, 0.5f),
		_bottomColor = glm::vec3(0.5, 0.5, 0.5);
	float _topColorIntensity = 1.0;
	float _middleColorIntensity = 1.0;
	float _bottomColorIntensity = 1.0;
	float _width = 1.0f, _height = 1.0f, _distance = 10.0f;
};

struct ViewerState {
	QString _volumeImageFile;
	std::vector<ChannelViewerState> _channels;
    int _resolutionX = 0, _resolutionY = 0;
    int _renderIterations = 1;
	float _exposure = 0.75f;
	float _densityScale = 50.0f;
	float _fov = 55.0f;
	float _apertureSize = 0.0f;
    float _focalDistance = 0.0f;
	float _gradientFactor = 50.0f;
	float _primaryStepSize = 1.0f, _secondaryStepSize = 1.0f;
	float _roiXmax = 1.0f, _roiYmax = 1.0f, _roiZmax = 1.0f, _roiXmin = 0.0f, _roiYmin = 0.0f, _roiZmin = 0.0f;
	float _scaleX = 1.0f, _scaleY = 1.0f, _scaleZ = 1.0f;

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
