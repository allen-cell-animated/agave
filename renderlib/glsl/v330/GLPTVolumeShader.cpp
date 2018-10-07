#include "glad/glad.h"
#include "GLPTVolumeShader.h"

#include <glm.h>
#include <gl/Util.h>

#include <iostream>
#include <sstream>


GLPTVolumeShader::GLPTVolumeShader():
    QOpenGLShaderProgram(),
    vshader(),
    fshader()
{
    vshader = new QOpenGLShader(QOpenGLShader::Vertex);
	vshader->compileSourceCode
	(R"(
#version 330 core

layout (location = 0) in vec2 position;
out VertexData
{
  vec2 pObj;
} outData;

void main()
{
  outData.pObj = position;
  gl_Position = vec4(position, 0.0, 1.0);
}
	)");

    if (!vshader->isCompiled())
    {
        std::cerr << "GLPTVolumeShader: Failed to compile vertex shader\n" << vshader->log().toStdString() << std::endl;
    }

    fshader = new QOpenGLShader(QOpenGLShader::Fragment);
	fshader->compileSourceCode
	(R"(
#version 330 core

in VertexData
{
  vec2 pObj;
} inData;

out vec4 outputColour;

uniform sampler3D textureVolume;

void main()
{
	outputColour = vec4(0.3, 0.0, 0.0, 1.0);
	return;
}
    )");

    if (!fshader->isCompiled())
    {
        std::cerr << "GLPTVolumeShader: Failed to compile fragment shader\n" << fshader->log().toStdString() << std::endl;
    }

    addShader(vshader);
    addShader(fshader);
    link();

    if (!isLinked())
    {
        std::cerr << "GLPTVolumeShader: Failed to link shader program\n" << log().toStdString() << std::endl;
    }
	
	uTextureVolume = uniformLocation("textureVolume");
}

GLPTVolumeShader::~GLPTVolumeShader()
{
}


void
GLPTVolumeShader::setShadingUniforms()
{
	glUniform1i(uTextureVolume, 0);


    for (int i = 0; i < nChannels; ++i) {
        if (_scene->_material.enabled[i] && activeChannel < MAX_GL_CHANNELS) {
            _renderBufferShader->lutTexture[activeChannel] = _imgCuda._channels[i]._volumeLutTexture;
            _renderBufferShader->intensityMax[activeChannel] = _scene->_volume->channel(i)->_max;
            _renderBufferShader->intensityMin[activeChannel] = _scene->_volume->channel(i)->_min;
            _renderBufferShader->diffuse[activeChannel * 3 + 0] = _scene->_material.diffuse[i * 3 + 0];
            _renderBufferShader->diffuse[activeChannel * 3 + 1] = _scene->_material.diffuse[i * 3 + 1];
            _renderBufferShader->diffuse[activeChannel * 3 + 2] = _scene->_material.diffuse[i * 3 + 2];
            _renderBufferShader->specular[activeChannel * 3 + 0] = _scene->_material.specular[i * 3 + 0];
            _renderBufferShader->specular[activeChannel * 3 + 1] = _scene->_material.specular[i * 3 + 1];
            _renderBufferShader->specular[activeChannel * 3 + 2] = _scene->_material.specular[i * 3 + 2];
            _renderBufferShader->emissive[activeChannel * 3 + 0] = _scene->_material.emissive[i * 3 + 0];
            _renderBufferShader->emissive[activeChannel * 3 + 1] = _scene->_material.emissive[i * 3 + 1];
            _renderBufferShader->emissive[activeChannel * 3 + 2] = _scene->_material.emissive[i * 3 + 2];
            _renderBufferShader->roughness[activeChannel] = _scene->_material.roughness[i];
            _renderBufferShader->opacity[activeChannel] = _scene->_material.opacity[i];

            activeChannel++;
            _renderBufferShader->nChannels = activeChannel;
        }
    }


}

void 
GLPTVolumeShader::setTransformUniforms(const CCamera& camera, const glm::mat4& modelMatrix)
{
	float w = (float)camera.m_Film.GetWidth();
	float h = (float)camera.m_Film.GetHeight();
	float vfov = camera.m_FovV * DEG_TO_RAD;

	glm::vec3 eye(camera.m_From.x, camera.m_From.y, camera.m_From.z);
	glm::vec3 center(camera.m_Target.x, camera.m_Target.y, camera.m_Target.z);
	glm::vec3 up(camera.m_Up.x, camera.m_Up.y, camera.m_Up.z);
	glm::mat4 cv = glm::lookAt(eye, center, up);
	glm::mat4 cp = glm::perspectiveFov(vfov, w, h, camera.m_Hither, camera.m_Yon);

	//glUniform3fv(uCameraPosition, 1, glm::value_ptr(camera.position));
	glUniformMatrix4fv(uProjectionMatrix, 1, GL_FALSE, glm::value_ptr(cp));
	glUniformMatrix4fv(uModelViewMatrix, 1, GL_FALSE, glm::value_ptr(cv * modelMatrix));
	glUniformMatrix4fv(uInverseModelViewMatrix, 1, GL_FALSE, glm::value_ptr(glm::inverse(cv * modelMatrix)));
}
