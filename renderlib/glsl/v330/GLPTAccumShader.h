#pragma once

#include <QOpenGLShaderProgram>

#include <glm.h>
#include "CCamera.h"

/**
    */
class GLPTAccumShader : public QOpenGLShaderProgram
{

public:
    /**
    * Constructor.
    *
    * @param parent the parent of this object.
    */
    explicit GLPTAccumShader();

    /// Destructor.
    ~GLPTAccumShader();

    /**
    * Set vertex coordinates from array.
    *
    * @param offset data offset if using a buffer object otherwise
    * the coordinate values.
    * @param tupleSize the tuple size of the data.
    * @param stride the stride of the data.
    */
    void
    setCoords(const GLfloat *offset = 0,
            int            tupleSize = 2,
            int            stride = 0);

    /**
    * Set vertex coordinates from buffer object.
    *
    * @param coords the coordinate values; null if using a buffer object.
    * @param offset the offset into the coords buffer.
    * @param tupleSize the tuple size of the data.
    * @param stride the stride of the data.
    */
    void
    setCoords(GLuint  coords,
            const GLfloat  *offset = 0,
            int             tupleSize = 2,
            int             stride = 0);



	void setTransformUniforms(const CCamera& camera, const glm::mat4& modelMatrix);
	void setShadingUniforms();

	float dataRangeMin;
	float dataRangeMax;
	float GAMMA_MIN;
	float GAMMA_MAX;
	float GAMMA_SCALE;
	float BRIGHTNESS;
	float DENSITY;
	float maskAlpha;
	int BREAK_STEPS;
	glm::vec3 AABB_CLIP_MIN;
	glm::vec3 AABB_CLIP_MAX;

private:
    /// The vertex shader.
    QOpenGLShader *vshader;
    /// The fragment shader.
    QOpenGLShader *fshader;

	int
		uModelViewMatrix,
		uProjectionMatrix,
		uDataRangeMin,
		uDataRangeMax,
		uBreakSteps,
		uAABBClipMin,
		uAABBClipMax,
		uInverseModelViewMatrix,
		uCameraPosition,
		uResolution,
		uGammaMin,
		uGammaMax,
		uGammaScale,
		uBrightness,
		uDensity,
		uMaskAlpha,
		uTextureAtlas,
		uTextureAtlasMask;
};

