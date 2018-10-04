#pragma once

#include <QOpenGLShaderProgram>

#include <glm.h>
#include "CCamera.h"

#define MAX_GL_CHANNELS 4

/**
    */
class GLPTVolumeShader : public QOpenGLShaderProgram
{

public:
    /**
    * Constructor.
    *
    * @param parent the parent of this object.
    */
    explicit GLPTVolumeShader();

    /// Destructor.
    ~GLPTVolumeShader();

	void setTransformUniforms(const CCamera& camera, const glm::mat4& modelMatrix);
	void setShadingUniforms();

    int nChannels;
    float intensityMax[MAX_GL_CHANNELS];
    float intensityMin[MAX_GL_CHANNELS];
    float diffuse[MAX_GL_CHANNELS * 3];
    float specular[MAX_GL_CHANNELS * 3];
    float emissive[MAX_GL_CHANNELS * 3];
    float roughness[MAX_GL_CHANNELS];
    float opacity[MAX_GL_CHANNELS];

    GLuint volumeTexture;
    GLuint lutTexture[MAX_GL_CHANNELS];


private:
    /// The vertex shader.
    QOpenGLShader *vshader;
    /// The fragment shader.
    QOpenGLShader *fshader;

	int
		uModelViewMatrix,
		uProjectionMatrix,
		uInverseModelViewMatrix,
		uCameraPosition,
		uResolution,
		uTextureVolume;
};

