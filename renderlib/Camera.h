#pragma once

#include "glm.h"

/**
* Camera (modelview projection matrix manipulation)
*/
class Camera
{
public:
	/// Projection type
	enum ProjectionType
	{
		ORTHOGRAPHIC, ///< Orthographic projection.
		PERSPECTIVE   ///< Perspective projection.
	};

	Camera() :
		projectionType(ORTHOGRAPHIC),
		zoom(0),
		xTran(0),
		yTran(0),
		zRot(0),
		model(1.0f),
		view(1.0f),
		projection(1.0f),
		position(0.0, 2.5, 0.0),
		direction(0.0, 0.0, -1.0),
		up(0.0, 1.0, 0.0)
	{}


	// speed is usually 0.1f or something small like that
	void rotate(float amount, glm::vec3& axis)
	{
		direction = glm::rotate(direction, amount, axis);
		up = glm::rotate(up, amount, axis);
	}

	void translate(glm::vec3& translation)
	{
		position += translation;
	}

	void update()
	{
		view = glm::lookAt(position, position + direction, up);
	}


	/// Projection type.
	ProjectionType projectionType;
	/// Zoom factor.
	int zoom;
	/// x translation
	int xTran;
	/// y translation.
	int yTran;
	/// Rotation factor.
	int zRot;

	glm::vec3 position;
	glm::vec3 direction;
	glm::vec3 up;

	/// Current model.
	glm::mat4 model;
	/// Current view.
	glm::mat4 view;
	/// Current projection.
	glm::mat4 projection;

	/**
	* Get zoom factor.
	*
	* Convert linear signed zoom value to a factor (to the 10th
	* power of the zoom value).
	*
	* @returns the zoom factor.
	*/
	float
		zoomfactor() const
	{
		return std::pow(10.0f, static_cast<float>(zoom) / 1024.0f); /// @todo remove fixed size.
	}

	/**
	* Get rotation factor.
	*
	* @returns the rotation factor (in radians).
	*/
	float
		rotation() const
	{
		return glm::radians(-static_cast<float>(zRot) / 16.0f);
	}

	/**
	* Get modelview projection matrix.
	*
	* The separate model, view and projection matrices are
	* combined to form a single matrix.
	*
	* @returns the modelview projection matrix.
	*/
	glm::mat4
		mvp() const
	{
		return projection * view * model;
	}

	glm::mat4
		mv() const
	{
		return view * model;
	}

	glm::mat4
		mv_inv() const
	{
		return glm::inverse (mv());
	}
};

