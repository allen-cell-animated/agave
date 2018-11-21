#include "ViewerState.h"

#include <QDebug>
#include <QFile>
#include <QJsonArray>
#include <QJsonObject>

QJsonArray jsonVec3(float x, float y, float z) {
	QJsonArray tgt;
	tgt.append(x);
	tgt.append(y);
	tgt.append(z);
	return tgt;
}
QJsonArray jsonVec2(float x, float y) {
	QJsonArray tgt;
	tgt.append(x);
	tgt.append(y);
	return tgt;
}
QJsonArray jsonVec3(const glm::vec3& v) {
	QJsonArray tgt;
	tgt.append(v.x);
	tgt.append(v.y);
	tgt.append(v.z);
	return tgt;
}
QJsonArray jsonVec2(const glm::vec2& v) {
	QJsonArray tgt;
	tgt.append(v.x);
	tgt.append(v.y);
	return tgt;
}

void getFloat(QJsonObject obj, QString prop, float& value) {
	if (obj.contains(prop)) {
		value = (float)obj[prop].toDouble(value);
	}
}
void getInt(QJsonObject obj, QString prop, int& value) {
	if (obj.contains(prop)) {
		value = obj[prop].toInt(value);
	}
}
void getString(QJsonObject obj, QString prop, QString& value) {
	if (obj.contains(prop)) {
		value = obj[prop].toString(value);
	}
}
void getBool(QJsonObject obj, QString prop, bool& value) {
	if (obj.contains(prop)) {
		value = obj[prop].toBool(value);
	}
}
void getVec3(QJsonObject obj, QString prop, glm::vec3& value) {
	if (obj.contains(prop)) {
		QJsonArray ja = obj[prop].toArray();
		value.x = ja.at(0).toDouble(value.x);
		value.y = ja.at(1).toDouble(value.y);
		value.z = ja.at(2).toDouble(value.z);
	}
}
void getVec2(QJsonObject obj, QString prop, glm::vec2& value) {
	if (obj.contains(prop)) {
		QJsonArray ja = obj[prop].toArray();
		value.x = ja.at(0).toDouble(value.x);
		value.y = ja.at(1).toDouble(value.y);
	}
}
void getVec2i(QJsonObject obj, QString prop, glm::ivec2& value) {
	if (obj.contains(prop)) {
		QJsonArray ja = obj[prop].toArray();
		value.x = ja.at(0).toInt(value.x);
		value.y = ja.at(1).toInt(value.y);
	}
}

ViewerState ViewerState::readStateFromJson(QString filePath) {
	// defaults from default ctor
	ViewerState p;

	// try to open server.cfg
	QFile loadFile(filePath);
	if (!loadFile.open(QIODevice::ReadOnly)) {
		qDebug() << "No config file found openable at " << filePath;
		return p;
	}

	QByteArray jsonData = loadFile.readAll();
	QJsonDocument jsonDoc(QJsonDocument::fromJson(jsonData));
	if (jsonDoc.isNull()) {
		qDebug() << "Invalid config file format. Make sure it is json.";
		return p;
	}

	p.stateFromJson(jsonDoc);

	return p;
}


void ViewerState::stateFromJson(QJsonDocument& jsonDoc)
{
	QJsonObject json(jsonDoc.object());

	getString(json, "name", _volumeImageFile);
	getInt(json, "renderIterations", _renderIterations);
	getFloat(json, "density", _densityScale);

	glm::ivec2 res(_resolutionX, _resolutionY);
	getVec2i(json, "resolution", res);
	_resolutionX = res.x;
	_resolutionY = res.y;

	glm::vec3 scale(_scaleX, _scaleY, _scaleZ);
	getVec3(json, "scale", scale);
	_scaleX = scale.x; _scaleY = scale.y; _scaleZ = scale.z;

	if (json.contains("clipRegion") && json["clipRegion"].isArray()) {
		QJsonArray ja = json["clipRegion"].toArray();
		QJsonArray crx = ja.at(0).toArray();
		_roiXmin = crx.at(0).toDouble(_roiXmin);
		_roiXmax = crx.at(1).toDouble(_roiXmax);
		QJsonArray cry = ja.at(1).toArray();
		_roiYmin = cry.at(0).toDouble(_roiYmin);
		_roiYmax = cry.at(1).toDouble(_roiYmax);
		QJsonArray crz = ja.at(2).toArray();
		_roiZmin = crz.at(0).toDouble(_roiZmin);
		_roiZmax = crz.at(1).toDouble(_roiZmax);
	}

	if (json.contains("camera") && json["camera"].isObject()) {
		QJsonObject cam = json["camera"].toObject();
		glm::vec3 tmp;
		getVec3(cam, "eye", tmp);
		_eyeX = tmp.x; _eyeY = tmp.y; _eyeZ = tmp.z;
		getVec3(cam, "target", tmp);
		_targetX = tmp.x; _targetY = tmp.y; _targetZ = tmp.z;
		getVec3(cam, "up", tmp);
		_upX = tmp.x; _upY = tmp.y; _upZ = tmp.z;
		getFloat(cam, "fovY", _fov);
		getFloat(cam, "exposure", _exposure);
		getFloat(cam, "aperture", _apertureSize);
		getFloat(cam, "focalDistance", _focalDistance);
	}

	if (json.contains("channels") && json["channels"].isArray()) {
		QJsonArray channelsArray = json["channels"].toArray();
		_channels.clear();
		_channels.reserve(channelsArray.size());
		for (int i = 0; i < channelsArray.size(); ++i) {
			ChannelViewerState ch;
			QJsonObject channeli = channelsArray[i].toObject();

			getBool(channeli, "enabled", ch._enabled);
			getVec3(channeli, "diffuseColor", ch._diffuse);
			getVec3(channeli, "specularColor", ch._specular);
			getVec3(channeli, "emissiveColor", ch._emissive);
			getFloat(channeli, "glossiness", ch._glossiness);
			getFloat(channeli, "window", ch._window);
			getFloat(channeli, "level", ch._level);

			QString channelsString = channelsArray[i].toString();
			_channels.push_back(ch);
		}
	}

	// lights
	if (json.contains("lights") && json["lights"].isArray()) {
		QJsonArray lightsArray = json["lights"].toArray();
		// expect two.
		for (int i = 0; i < std::min(lightsArray.size(), 2); ++i) {
			LightViewerState& ls = (i == 0) ? _light0 : _light1;
			QJsonObject lighti = lightsArray[i].toObject();
			getInt(lighti, "type", ls._type);
			getVec3(lighti, "topColor", ls._topColor);
			getVec3(lighti, "middleColor", ls._middleColor);
			getVec3(lighti, "color", ls._color);
			getVec3(lighti, "bottomColor", ls._bottomColor);
			getFloat(lighti, "distance", ls._distance);
			getFloat(lighti, "theta", ls._theta);
			getFloat(lighti, "phi", ls._phi);
			getFloat(lighti, "width", ls._width);
			getFloat(lighti, "height", ls._height);
		}
	}
}

QJsonDocument ViewerState::stateToJson() const
{
	// fire back some json...
	QJsonObject j;
	j["name"] = _volumeImageFile;
	
	QJsonArray resolution;
	resolution.append(_resolutionX);
	resolution.append(_resolutionY);
	j["resolution"] = resolution;

	j["renderIterations"] = _renderIterations;

	QJsonArray clipRegion;
	QJsonArray clipRegionX;
	clipRegionX.append(_roiXmin);
	clipRegionX.append(_roiXmax);
	QJsonArray clipRegionY;
	clipRegionY.append(_roiYmin);
	clipRegionY.append(_roiYmax);
	QJsonArray clipRegionZ;
	clipRegionZ.append(_roiZmin);
	clipRegionZ.append(_roiZmax);
	clipRegion.append(clipRegionX);
	clipRegion.append(clipRegionY);
	clipRegion.append(clipRegionZ);

	j["clipRegion"] = clipRegion;

	j["scale"] = jsonVec3(_scaleX, _scaleY, _scaleZ);

	QJsonObject camera;
	camera["eye"] = jsonVec3(_eyeX, _eyeY, _eyeZ);
	camera["target"] = jsonVec3(_targetX, _targetY, _targetZ);
	camera["up"] = jsonVec3(_upX, _upY, _upZ);

	camera["fovY"] = _fov;

	camera["exposure"] = _exposure;
	camera["aperture"] = _apertureSize;
	camera["focalDistance"] = _focalDistance;
	j["camera"] = camera;

	QJsonArray channels;
	for (auto ch : _channels) {
		QJsonObject channel;
		channel["enabled"] = ch._enabled;
		channel["diffuseColor"] = jsonVec3(ch._diffuse.x, ch._diffuse.y, ch._diffuse.z);
		channel["specularColor"] = jsonVec3(ch._specular.x, ch._specular.y, ch._specular.z);
		channel["emissiveColor"] = jsonVec3(ch._emissive.x, ch._emissive.y, ch._emissive.z);
		channel["glossiness"] = ch._glossiness;
		channel["window"] = ch._window;
		channel["level"] = ch._level;

		channels.append(channel);
	}
	j["channels"] = channels;

	j["density"] = _densityScale;

	// lighting
	QJsonArray lights;
	QJsonObject light0;
	light0["type"] = _light0._type;
	light0["distance"] = _light0._distance;
	light0["theta"] = _light0._theta;
	light0["phi"] = _light0._phi;
	light0["color"] = jsonVec3(_light0._color.r, _light0._color.g, _light0._color.b);
	light0["topColor"] = jsonVec3(
		_light0._topColor.r, _light0._topColor.g, _light0._topColor.b
	);
	light0["middleColor"] = jsonVec3(
		_light0._middleColor.r, _light0._middleColor.g, _light0._middleColor.b
	);
	light0["bottomColor"] = jsonVec3(
		_light0._bottomColor.r, _light0._bottomColor.g, _light0._bottomColor.b
	);
	light0["width"] = _light0._width;
	light0["height"] = _light0._height;
	lights.append(light0);

	QJsonObject light1;
	light1["type"] = _light1._type;
	light1["distance"] = _light1._distance;
	light1["theta"] = _light1._theta;
	light1["phi"] = _light1._phi;
	light1["color"] = jsonVec3(_light1._color.r, _light1._color.g, _light1._color.b);
	light1["topColor"] = jsonVec3(
		_light1._topColor.r, _light1._topColor.g, _light1._topColor.b
	);
	light1["middleColor"] = jsonVec3(
		_light1._middleColor.r, _light1._middleColor.g, _light1._middleColor.b
	);
	light1["bottomColor"] = jsonVec3(
		_light1._bottomColor.r, _light1._bottomColor.g, _light1._bottomColor.b
	);
	light1["width"] = _light1._width;
	light1["height"] = _light1._height;
	lights.append(light1);
	j["lights"] = lights;

	return QJsonDocument(j);
}

void ViewerState::writeStateToJson(QString filePath, const ViewerState& state)
{
    QJsonDocument d = state.stateToJson();
}
