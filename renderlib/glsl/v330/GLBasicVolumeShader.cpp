#include "glad/glad.h"
#include "GLBasicVolumeShader.h"

#include <glm.h>
#include <gl/Util.h>

#include <iostream>
#include <sstream>


GLBasicVolumeShader::GLBasicVolumeShader():
    QOpenGLShaderProgram(),
    vshader(),
    fshader(),
    attr_coords()
{
    vshader = new QOpenGLShader(QOpenGLShader::Vertex);
	vshader->compileSourceCode
	(R"(
#version 330 core

layout (location = 0) in vec3 position;
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
out VertexData
{
  vec3 pObj;
} outData;

void main()
{
  outData.pObj = position;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
	)");

    if (!vshader->isCompiled())
    {
        std::cerr << "GLBasicVolumeShader: Failed to compile vertex shader\n" << vshader->log().toStdString() << std::endl;
    }

    fshader = new QOpenGLShader(QOpenGLShader::Fragment);
	fshader->compileSourceCode
	(R"(
#version 330 core

in VertexData
{
  vec3 pObj;
} inData;

out vec4 outputColour;

uniform vec3 cameraPosition;
uniform mat4 inverseModelMatrix;

uniform sampler3D textureAtlas;
uniform sampler2D textureAtlasMask;
//uniform sampler2D lut;

#define M_PI 3.14159265358979323846
uniform float GAMMA_MIN;
uniform float GAMMA_MAX;
uniform float GAMMA_SCALE;
uniform float BRIGHTNESS;
uniform float DENSITY;
uniform float maskAlpha;
uniform int BREAK_STEPS;
uniform vec3 AABB_CLIP_MIN;
uniform vec3 AABB_CLIP_MAX;

uniform float dataRangeMin; // 0..1 (mapped from 0..uint16_max)
uniform float dataRangeMax; // 0..1 (mapped from 0..uint16_max)

float powf(float a, float b) {
	return pow(a, b);
}

float rand(vec2 co) {
	float threadId = gl_FragCoord.x / (gl_FragCoord.y + 1.0);
	float bigVal = threadId*1299721.0 / 911.0;
	vec2 smallVal = vec2(threadId*7927.0 / 577.0, threadId*104743.0 / 1039.0);
	return fract(sin(dot(co, smallVal)) * bigVal);
}

vec4 luma2Alpha(vec4 color, float vmin, float vmax, float C) {
	float x = max(color[2], max(color[0], color[1]));
	float xi = (x - vmin) / (vmax - vmin);
	xi = clamp(xi, 0.0, 1.0);
	float y = pow(xi, C);
	y = clamp(y, 0.0, 1.0);
	color[3] = y;
	return(color);
}

vec4 sampleAs3DTexture(sampler3D tex, vec4 pos) {
	float bounds = float(pos[0] > 0.001 && pos[0] < 0.999 &&
		pos[1] > 0.001 && pos[1] < 0.999 &&
		pos[2] > 0.001 && pos[2] < 0.999);

    float texval = texture(tex, pos.xyz).r;
	texval = (texval - dataRangeMin) / (dataRangeMax - dataRangeMin);
	vec4 retval = vec4(texval, texval, texval, 1.0);
	return bounds*retval;
}

vec4 sampleStack(sampler3D tex, vec4 pos) {
	vec4 col = sampleAs3DTexture(tex, pos);
	col = luma2Alpha(col, GAMMA_MIN, GAMMA_MAX, GAMMA_SCALE);
	return col;
}

//->intersect AXIS-ALIGNED box routine
//
bool intersectBox(in vec3 r_o, in vec3 r_d, in vec3 boxMin, in vec3 boxMax, out float tnear, out float tfar) {
	vec3 invR = vec3(1.0, 1.0, 1.0) / r_d;
	vec3 tbot = invR * (boxMin - r_o);
	vec3 ttop = invR * (boxMax - r_o);
	vec3 tmin = min(ttop, tbot);
	vec3 tmax = max(ttop, tbot);
	float largest_tmin = max(max(tmin.x, tmin.y), max(tmin.x, tmin.z));
	float smallest_tmax = min(min(tmax.x, tmax.y), min(tmax.x, tmax.z));
	tnear = largest_tmin;
	tfar = smallest_tmax;
	return(smallest_tmax > largest_tmin);
}

vec4 integrateVolume(vec4 eye_o, vec4 eye_d,
	float tnear, float tfar,
	float clipNear, float clipFar,
	sampler3D textureAtlas
) {
	vec4 C = vec4(0.0);
	float tend = tfar;
	float tbegin = tnear;
	const int maxSteps = 512;
	float csteps = float(BREAK_STEPS);
	csteps = clamp(csteps, 0.0, float(maxSteps));
	float isteps = 1.0 / csteps;
	float r = 0.5 - 1.0*rand(eye_d.xy);
	float tstep = isteps / length(eye_d);
	float tfarsurf = /*float(DITHERING)**/r*tstep;
	float overflow = mod((tfarsurf - tend), tstep);
	float t = tbegin + overflow;
	t += /*float(DITHERING)**/r*tstep;
	float tdist = 0.0;
	int numSteps = 0;

	vec4 pos, col;
	for (int i = 0; i<maxSteps; i++) {
		pos = eye_o + eye_d*t;
		pos.xyz = (pos.xyz - AABB_CLIP_MIN) / (AABB_CLIP_MAX - AABB_CLIP_MIN);//0.5 * (pos + 1.0); // map position from [boxMin, boxMax] to [0, 1] coordinates
		col = sampleStack(textureAtlas, pos);

		//Finish up by adding brightness/density
		col.xyz *= BRIGHTNESS;
		col.w *= DENSITY;
		float s = 0.5*float(256) / float(BREAK_STEPS);
		float stepScale = (1.0 - powf((1.0 - col.w), s));
		col.w = stepScale;
		col.xyz *= col.w;
		col = clamp(col, 0.0, 1.0);

		C = (1.0 - C.w)*col + C;
		t += tstep;
		numSteps = i;
		if (i > BREAK_STEPS || t  > tend || t > tbegin + clipFar) 
			break;
		if (C.w > 1.0) 
			break;
	}
	return C;
}
void main()
{
	outputColour = vec4(1.0, 0.0, 0.0, 1.0);

	vec3 eyeRay_o = (inverseModelMatrix * vec4(cameraPosition, 1.0)).xyz;
	vec3 eyeRay_d = inData.pObj - eyeRay_o;
	vec3 boxMin = AABB_CLIP_MIN;
	vec3 boxMax = AABB_CLIP_MAX;
	float tnear, tfar;
	bool hit = intersectBox(eyeRay_o, eyeRay_d, boxMin, boxMax, tnear, tfar);
	if (!hit) {
		outputColour = vec4(1.0, 0.0, 1.0, 1.0);
		return;
	}
	float clipNear = 0.0;//-(dot(eyeRay_o.xyz, eyeNorm) + dNear) / dot(eyeRay_d.xyz, eyeNorm);
	float clipFar = 10000.0;//-(dot(eyeRay_o.xyz,-eyeNorm) + dFar ) / dot(eyeRay_d.xyz,-eyeNorm);

	vec4 C = integrateVolume(vec4(eyeRay_o, 1.0), vec4(eyeRay_d, 0.0),
		tnear, tfar,
		clipNear, clipFar,
		textureAtlas);
	C = clamp(C, 0.0, 1.0);
	outputColour = C;
	return;
}
    )");

    if (!fshader->isCompiled())
    {
        std::cerr << "GLBasicVolumeShader: Failed to compile fragment shader\n" << fshader->log().toStdString() << std::endl;
    }

    addShader(vshader);
    addShader(fshader);
    link();

    if (!isLinked())
    {
        std::cerr << "GLBasicVolumeShader: Failed to link shader program\n" << log().toStdString() << std::endl;
    }

    attr_coords = attributeLocation("position");
    if (attr_coords == -1)
    std::cerr << "GLBasicVolumeShader: Failed to bind coordinates" << std::endl;
	
	uModelViewMatrix = uniformLocation("modelViewMatrix");
	uProjectionMatrix = uniformLocation("projectionMatrix");

	uBreakSteps = uniformLocation("BREAK_STEPS");
	uAABBClipMin = uniformLocation("AABB_CLIP_MIN");
	uAABBClipMax = uniformLocation("AABB_CLIP_MAX");
	uInverseModelMatrix = uniformLocation("inverseModelMatrix");
	uCameraPosition = uniformLocation("cameraPosition");
	uGammaMin = uniformLocation("GAMMA_MIN");
	uGammaMax = uniformLocation("GAMMA_MAX");
	uGammaScale = uniformLocation("GAMMA_SCALE");
	uBrightness = uniformLocation("BRIGHTNESS");
	uDensity = uniformLocation("DENSITY");
	uMaskAlpha = uniformLocation("maskAlpha");
	
	uTextureAtlas = uniformLocation("textureAtlas");
	uTextureAtlasMask = uniformLocation("textureAtlasMask");

	uDataRangeMin = uniformLocation("dataRangeMin");
	uDataRangeMax = uniformLocation("dataRangeMax");
}

GLBasicVolumeShader::~GLBasicVolumeShader()
{
}

void
GLBasicVolumeShader::enableCoords()
{
    enableAttributeArray(attr_coords);
}

void
GLBasicVolumeShader::disableCoords()
{
    disableAttributeArray(attr_coords);
}

void
GLBasicVolumeShader::setCoords(const GLfloat *offset, int tupleSize, int stride)
{
    setAttributeArray(attr_coords, offset, tupleSize, stride);
}

void
GLBasicVolumeShader::setCoords(GLuint coords, const GLfloat *offset, int tupleSize, int stride)
{
	glBindBuffer(GL_ARRAY_BUFFER, coords);
    setCoords(offset, tupleSize, stride);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void
GLBasicVolumeShader::setTexture(int texunit)
{
    glUniform1i(uTextureAtlas, texunit);
    check_gl("Set image texture");
}

void
GLBasicVolumeShader::setCorrection(const glm::vec3& corr)
{
//    glUniform3fv(uniform_corr, 1, glm::value_ptr(corr));
    check_gl("Set correction multiplier");
}

void
GLBasicVolumeShader::setLUT(int texunit)
{
 //   glUniform1i(uniform_lut, texunit);
    check_gl("Set LUT texture");
}

void
GLBasicVolumeShader::setShadingUniforms()
{
	glUniform1f(uDataRangeMin, dataRangeMin);
	glUniform1f(uDataRangeMax, dataRangeMax);
	glUniform1f(uGammaMin, GAMMA_MIN);
	glUniform1f(uGammaMax, GAMMA_MAX);
	glUniform1f(uGammaScale, GAMMA_SCALE);
	glUniform1f(uBrightness, BRIGHTNESS);
	glUniform1f(uDensity, DENSITY);
	glUniform1f(uMaskAlpha, maskAlpha);
	glUniform1i(uBreakSteps, BREAK_STEPS);
	glUniform3fv(uAABBClipMin, 1, glm::value_ptr(AABB_CLIP_MIN));
	glUniform3fv(uAABBClipMax, 1, glm::value_ptr(AABB_CLIP_MAX));
}
void 
GLBasicVolumeShader::setTransformUniforms(const Camera& camera, glm::mat4& modelMatrix)
{
	glUniform3fv(uCameraPosition, 1, glm::value_ptr(camera.position));
	glUniformMatrix4fv(uProjectionMatrix, 1, GL_FALSE, glm::value_ptr(camera.projection));
	glUniformMatrix4fv(uModelViewMatrix, 1, GL_FALSE, glm::value_ptr(camera.view * camera.model * modelMatrix));
	glUniformMatrix4fv(uInverseModelMatrix, 1, GL_FALSE, glm::value_ptr(glm::inverse(camera.model * modelMatrix)));
}
