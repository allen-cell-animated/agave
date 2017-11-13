#pragma once

#include "glad/glad.h"
#include "glm.h"
#include <string>
#include <vector>

class OpenGLShaderPrivate;
class OpenGLShader
{
public:
	enum ShaderType
	{
		Vertex = 0x0001,
		Fragment = 0x0002,
		Geometry = 0x0004,
		TessellationControl = 0x0008,
		TessellationEvaluation = 0x0010,
		Compute = 0x0020
	};

	explicit OpenGLShader(OpenGLShader::ShaderType type);
	virtual ~OpenGLShader();

	OpenGLShader::ShaderType shaderType() const;

	bool compileSourceCode(const char *source);
	bool compileSourceFile(const std::string& fileName);

	std::string sourceCode() const;

	bool isCompiled() const;
	std::string log() const;

	GLuint shaderId() const;

	static bool hasOpenGLShaders(ShaderType type);

private:
	friend class OpenGLShaderProgram;

	OpenGLShaderPrivate* d;
};

class OpenGLShaderProgramPrivate;
class OpenGLShaderProgram
{
public:
	explicit OpenGLShaderProgram();
	virtual ~OpenGLShaderProgram();

	bool addShader(OpenGLShader *shader);
	void removeShader(OpenGLShader *shader);
	std::vector<OpenGLShader *> shaders() const;

	bool addShaderFromSourceCode(OpenGLShader::ShaderType type, const char *source);
	bool addShaderFromSourceFile(OpenGLShader::ShaderType type, const std::string& fileName);

	void removeAllShaders();

	virtual bool link();
	bool isLinked() const;
	std::string log() const;

	bool bind();
	void release();

	bool create();

	GLuint programId() const;

	int maxGeometryOutputVertices() const;

	void setPatchVertexCount(int count);
	int patchVertexCount() const;

	void setDefaultOuterTessellationLevels(const std::vector<float> &levels);
	std::vector<float> defaultOuterTessellationLevels() const;

	void setDefaultInnerTessellationLevels(const std::vector<float> &levels);
	std::vector<float> defaultInnerTessellationLevels() const;

	void bindAttributeLocation(const char *name, int location);

	int attributeLocation(const char *name) const;

	void setAttributeValue(int location, GLfloat value);
	void setAttributeValue(int location, GLfloat x, GLfloat y);
	void setAttributeValue(int location, GLfloat x, GLfloat y, GLfloat z);
	void setAttributeValue(int location, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
	void setAttributeValue(int location, const glm::vec2& value);
	void setAttributeValue(int location, const glm::vec3& value);
	void setAttributeValue(int location, const glm::vec4& value);
	void setAttributeValue(int location, const GLfloat *values, int columns, int rows);

	void setAttributeValue(const char *name, GLfloat value);
	void setAttributeValue(const char *name, GLfloat x, GLfloat y);
	void setAttributeValue(const char *name, GLfloat x, GLfloat y, GLfloat z);
	void setAttributeValue(const char *name, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
	void setAttributeValue(const char *name, const glm::vec2& value);
	void setAttributeValue(const char *name, const glm::vec3& value);
	void setAttributeValue(const char *name, const glm::vec4& value);
	void setAttributeValue(const char *name, const GLfloat *values, int columns, int rows);

	void setAttributeArray
	(int location, const GLfloat *values, int tupleSize, int stride = 0);
	void setAttributeArray
	(int location, const glm::vec2 *values, int stride = 0);
	void setAttributeArray
	(int location, const glm::vec3 *values, int stride = 0);
	void setAttributeArray
	(int location, const glm::vec4 *values, int stride = 0);
	void setAttributeArray
	(int location, GLenum type, const void *values, int tupleSize, int stride = 0);
	void setAttributeArray
	(const char *name, const GLfloat *values, int tupleSize, int stride = 0);
	void setAttributeArray
	(const char *name, const glm::vec2 *values, int stride = 0);
	void setAttributeArray
	(const char *name, const glm::vec3 *values, int stride = 0);
	void setAttributeArray
	(const char *name, const glm::vec4 *values, int stride = 0);
	void setAttributeArray
	(const char *name, GLenum type, const void *values, int tupleSize, int stride = 0);

	void setAttributeBuffer
	(int location, GLenum type, int offset, int tupleSize, int stride = 0);
	void setAttributeBuffer
	(const char *name, GLenum type, int offset, int tupleSize, int stride = 0);

	void enableAttributeArray(int location);
	void enableAttributeArray(const char *name);
	void disableAttributeArray(int location);
	void disableAttributeArray(const char *name);

	int uniformLocation(const char *name) const;

	void setUniformValue(int location, GLfloat value);
	void setUniformValue(int location, GLint value);
	void setUniformValue(int location, GLuint value);
	void setUniformValue(int location, GLfloat x, GLfloat y);
	void setUniformValue(int location, GLfloat x, GLfloat y, GLfloat z);
	void setUniformValue(int location, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
	void setUniformValue(int location, const glm::vec2& value);
	void setUniformValue(int location, const glm::vec3& value);
	void setUniformValue(int location, const glm::vec4& value);
	void setUniformValue(int location, const glm::mat2& value);
	void setUniformValue(int location, const glm::mat3& value);
	void setUniformValue(int location, const glm::mat4& value);
	void setUniformValue(int location, const GLfloat value[2][2]);
	void setUniformValue(int location, const GLfloat value[3][3]);
	void setUniformValue(int location, const GLfloat value[4][4]);

	void setUniformValue(const char *name, GLfloat value);
	void setUniformValue(const char *name, GLint value);
	void setUniformValue(const char *name, GLuint value);
	void setUniformValue(const char *name, GLfloat x, GLfloat y);
	void setUniformValue(const char *name, GLfloat x, GLfloat y, GLfloat z);
	void setUniformValue(const char *name, GLfloat x, GLfloat y, GLfloat z, GLfloat w);
	void setUniformValue(const char *name, const glm::vec2& value);
	void setUniformValue(const char *name, const glm::vec3& value);
	void setUniformValue(const char *name, const glm::vec4& value);
	void setUniformValue(const char *name, const glm::mat2& value);
	void setUniformValue(const char *name, const glm::mat3& value);
	void setUniformValue(const char *name, const glm::mat4& value);
	void setUniformValue(const char *name, const GLfloat value[2][2]);
	void setUniformValue(const char *name, const GLfloat value[3][3]);
	void setUniformValue(const char *name, const GLfloat value[4][4]);

	void setUniformValueArray(int location, const GLfloat *values, int count, int tupleSize);
	void setUniformValueArray(int location, const GLint *values, int count);
	void setUniformValueArray(int location, const GLuint *values, int count);
	void setUniformValueArray(int location, const glm::vec2 *values, int count);
	void setUniformValueArray(int location, const glm::vec3 *values, int count);
	void setUniformValueArray(int location, const glm::vec4 *values, int count);
	void setUniformValueArray(int location, const glm::mat2 *values, int count);
	void setUniformValueArray(int location, const glm::mat3 *values, int count);
	void setUniformValueArray(int location, const glm::mat4 *values, int count);

	void setUniformValueArray(const char *name, const GLfloat *values, int count, int tupleSize);
	void setUniformValueArray(const char *name, const GLint *values, int count);
	void setUniformValueArray(const char *name, const GLuint *values, int count);
	void setUniformValueArray(const char *name, const glm::vec2 *values, int count);
	void setUniformValueArray(const char *name, const glm::vec3 *values, int count);
	void setUniformValueArray(const char *name, const glm::vec4 *values, int count);
	void setUniformValueArray(const char *name, const glm::mat2 *values, int count);
	void setUniformValueArray(const char *name, const glm::mat3 *values, int count);
	void setUniformValueArray(const char *name, const glm::mat4 *values, int count);

	void shaderDestroyed();

private:

		bool init();

		OpenGLShaderProgramPrivate* d;
};

