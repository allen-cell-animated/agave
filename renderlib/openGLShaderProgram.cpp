#include "openglshaderprogram.h"
//#include "qopenglprogrambinarycache_p.h"

#include "Logging.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

/*!
\class OpenGLShaderProgram
\brief The OpenGLShaderProgram class allows OpenGL shader programs to be linked and used.
\since 5.0
\ingroup painting-3D
\inmodule QtGui

\section1 Introduction

This class supports shader programs written in the OpenGL Shading
Language (GLSL) and in the OpenGL/ES Shading Language (GLSL/ES).

OpenGLShader and OpenGLShaderProgram shelter the programmer from the details of
compiling and linking vertex and fragment shaders.

The following example creates a vertex shader program using the
supplied source \c{code}.  Once compiled and linked, the shader
program is activated in the current QOpenGLContext by calling
OpenGLShaderProgram::bind():

\snippet code/src_gui_qopenglshaderprogram.cpp 0

\section1 Writing Portable Shaders

Shader programs can be difficult to reuse across OpenGL implementations
because of varying levels of support for standard vertex attributes and
uniform variables.  In particular, GLSL/ES lacks all of the
standard variables that are present on desktop OpenGL systems:
\c{gl_Vertex}, \c{gl_Normal}, \c{gl_Color}, and so on.  Desktop OpenGL
lacks the variable qualifiers \c{highp}, \c{mediump}, and \c{lowp}.

The OpenGLShaderProgram class makes the process of writing portable shaders
easier by prefixing all shader programs with the following lines on
desktop OpenGL:

\code
#define highp
#define mediump
#define lowp
\endcode

This makes it possible to run most GLSL/ES shader programs
on desktop systems.  The programmer should restrict themselves
to just features that are present in GLSL/ES, and avoid
standard variable names that only work on the desktop.

\section1 Simple Shader Example

\snippet code/src_gui_qopenglshaderprogram.cpp 1

With the above shader program active, we can draw a green triangle
as follows:

\snippet code/src_gui_qopenglshaderprogram.cpp 2

\section1 Binary Shaders and Programs

Binary shaders may be specified using \c{glShaderBinary()} on
the return value from OpenGLShader::shaderId().  The OpenGLShader instance
containing the binary can then be added to the shader program with
addShader() and linked in the usual fashion with link().

Binary programs may be specified using \c{glProgramBinaryOES()}
on the return value from programId().  Then the application should
call link(), which will notice that the program has already been
specified and linked, allowing other operations to be performed
on the shader program. The shader program's id can be explicitly
created using the create() function.

\section2 Caching Program Binaries

As of Qt 5.9, support for caching program binaries on disk is built in. To
enable this, switch to using addCacheableShaderFromSourceCode() and
addCacheableShaderFromSourceFile(). With an OpenGL ES 3.x context or support
for \c{GL_ARB_get_program_binary}, this will transparently cache program
binaries under QStandardPaths::GenericCacheLocation or
QStandardPaths::CacheLocation. When support is not available, calling the
cacheable function variants is equivalent to the normal ones.

\note Some drivers do not have any binary formats available, even though
they advertise the extension or offer OpenGL ES 3.0. In this case program
binary support will be disabled.

\sa OpenGLShader
*/

/*!
\class OpenGLShader
\brief The OpenGLShader class allows OpenGL shaders to be compiled.
\since 5.0
\ingroup painting-3D
\inmodule QtGui

This class supports shaders written in the OpenGL Shading Language (GLSL)
and in the OpenGL/ES Shading Language (GLSL/ES).

OpenGLShader and OpenGLShaderProgram shelter the programmer from the details of
compiling and linking vertex and fragment shaders.

\sa OpenGLShaderProgram
*/

/*!
\enum OpenGLShader::ShaderTypeBit
This enum specifies the type of OpenGLShader that is being created.

\value Vertex Vertex shader written in the OpenGL Shading Language (GLSL).
\value Fragment Fragment shader written in the OpenGL Shading Language (GLSL).
\value Geometry Geometry shaders written in the OpenGL Shading Language (GLSL)
based on the OpenGL core feature (requires OpenGL >= 3.2).
\value TessellationControl Tessellation control shaders written in the OpenGL
shading language (GLSL), based on the core feature (requires OpenGL >= 4.0).
\value TessellationEvaluation Tessellation evaluation shaders written in the OpenGL
shading language (GLSL), based on the core feature (requires OpenGL >= 4.0).
\value Compute Compute shaders written in the OpenGL shading language (GLSL),
based on the core feature (requires OpenGL >= 4.3).
*/



static inline bool supportsGeometry()
{
	return true; // >=3.2
}

static inline bool supportsCompute()
{
	return true; // >= 4.3
}

static inline bool supportsTessellation()
{
	return true; // >= 4.0
}

class OpenGLSharedResourceGuard
{
public:
	typedef void(*FreeResourceFunc)(GLuint id);
	OpenGLSharedResourceGuard(GLuint id, FreeResourceFunc func)
		: m_id(id)
		, m_func(func)
	{
	}

	GLuint id() const { return m_id; }

protected:
	void invalidateResource()
	{
		m_id = 0;
	}

	void freeResource() 
	{
		if (m_id) {
			m_func(m_id);
			m_id = 0;
		}
	}

private:
	GLuint m_id;
	FreeResourceFunc m_func;
};

class OpenGLShaderPrivate
{
public:
	OpenGLShaderPrivate(OpenGLShader::ShaderType type)
		: shaderGuard(0)
		, shaderType(type)
		, compiled(false)
		, supportsGeometryShaders(false)
		, supportsTessellationShaders(false)
		, supportsComputeShaders(false)
	{
		if (shaderType & OpenGLShader::Geometry)
			supportsGeometryShaders = supportsGeometry();
		else if (shaderType & (OpenGLShader::TessellationControl | OpenGLShader::TessellationEvaluation))
			supportsTessellationShaders = supportsTessellation();
		else if (shaderType & OpenGLShader::Compute)
			supportsComputeShaders = supportsCompute();
	}
	~OpenGLShaderPrivate();

	OpenGLSharedResourceGuard *shaderGuard;
	OpenGLShader::ShaderType shaderType;
	bool compiled;
	std::string log;

	// Support for geometry shaders
	bool supportsGeometryShaders;
	// Support for tessellation shaders
	bool supportsTessellationShaders;
	// Support for compute shaders
	bool supportsComputeShaders;


	bool create();
	bool compile(OpenGLShader *q);
	void deleteShader();
};

namespace {
	void freeShaderFunc(GLuint id)
	{
		glDeleteShader(id);
	}
}

OpenGLShaderPrivate::~OpenGLShaderPrivate()
{
	if (shaderGuard) {
		delete shaderGuard;
		shaderGuard = nullptr;
	}
}

bool OpenGLShaderPrivate::create()
{
	GLuint shader = 0;
	if (shaderType == OpenGLShader::Vertex) {
		shader = glCreateShader(GL_VERTEX_SHADER);
	}
	else if (shaderType == OpenGLShader::Geometry && supportsGeometryShaders) {
		shader = glCreateShader(GL_GEOMETRY_SHADER);
	}
	else if (shaderType == OpenGLShader::TessellationControl && supportsTessellationShaders) {
		shader = glCreateShader(GL_TESS_CONTROL_SHADER);
	}
	else if (shaderType == OpenGLShader::TessellationEvaluation && supportsTessellationShaders) {
		shader = glCreateShader(GL_TESS_EVALUATION_SHADER);
	}
	else if (shaderType == OpenGLShader::Compute && supportsComputeShaders) {
		shader = glCreateShader(GL_COMPUTE_SHADER);
	}
	else if (shaderType == OpenGLShader::Fragment) {
		shader = glCreateShader(GL_FRAGMENT_SHADER);
	}
	if (!shader) {
		LOG_WARNING << "OpenGLShader: could not create shader";
		return false;
	}
	shaderGuard = new OpenGLSharedResourceGuard(shader, freeShaderFunc);
	return true;
}

bool OpenGLShaderPrivate::compile(OpenGLShader *q)
{
	GLuint shader = shaderGuard ? shaderGuard->id() : 0;
	if (!shader)
		return false;

	// Try to compile shader
	glCompileShader(shader);
	GLint value = 0;

	// Get compilation status
	glGetShaderiv(shader, GL_COMPILE_STATUS, &value);
	compiled = (value != 0);

	if (!compiled) {
		// Compilation failed, try to provide some information about the failure

		const char *types[] = {
			"Fragment",
			"Vertex",
			"Geometry",
			"Tessellation Control",
			"Tessellation Evaluation",
			"Compute",
			""
		};

		const char *type = types[6];
		switch (shaderType) {
		case OpenGLShader::Fragment:
			type = types[0]; break;
		case OpenGLShader::Vertex:
			type = types[1]; break;
		case OpenGLShader::Geometry:
			type = types[2]; break;
		case OpenGLShader::TessellationControl:
			type = types[3]; break;
		case OpenGLShader::TessellationEvaluation:
			type = types[4]; break;
		case OpenGLShader::Compute:
			type = types[5]; break;
		}

		// Get info and source code lengths
		GLint infoLogLength = 0;
		GLint sourceCodeLength = 0;
		char *logBuffer = 0;
		char *sourceCodeBuffer = 0;

		// Get the compilation info log
		glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLogLength);

		if (infoLogLength > 1) {
			GLint temp;
			logBuffer = new char[infoLogLength];
			glGetShaderInfoLog(shader, infoLogLength, &temp, logBuffer);
		}

		// Get the source code
		glGetShaderiv(shader, GL_SHADER_SOURCE_LENGTH, &sourceCodeLength);

		if (sourceCodeLength > 1) {
			GLint temp;
			sourceCodeBuffer = new char[sourceCodeLength];
			glGetShaderSource(shader, sourceCodeLength, &temp, sourceCodeBuffer);
		}

		if (logBuffer)
			log = std::string(logBuffer);
		else
			log = std::string("failed");

		LOG_WARNING << "OpenGLShader::compile(" << type << "): " << log;

		// Dump the source code if we got it
		if (sourceCodeBuffer) {
			LOG_WARNING << "*** Problematic " << type << " shader source code ***\n" << sourceCodeBuffer << "\n***";
		}

		// Cleanup
		delete[] logBuffer;
		delete[] sourceCodeBuffer;
	}

	return compiled;
}

void OpenGLShaderPrivate::deleteShader()
{
	if (shaderGuard) {
		delete shaderGuard;
		shaderGuard = 0;
	}
}

/*!
Constructs a new OpenGLShader object of the specified \a type
and attaches it to \a parent.  If shader programs are not supported,
OpenGLShaderProgram::hasOpenGLShaderPrograms() will return false.

This constructor is normally followed by a call to compileSourceCode()
or compileSourceFile().

The shader will be associated with the current QOpenGLContext.

\sa compileSourceCode(), compileSourceFile()
*/
OpenGLShader::OpenGLShader(OpenGLShader::ShaderType type)
{
	d = new OpenGLShaderPrivate(type);
	d->create();
}

/*!
Deletes this shader.  If the shader has been attached to a
OpenGLShaderProgram object, then the actual shader will stay around
until the OpenGLShaderProgram is destroyed.
*/
OpenGLShader::~OpenGLShader()
{
	delete d;
}

/*!
Returns the type of this shader.
*/
OpenGLShader::ShaderType OpenGLShader::shaderType() const
{
	return d->shaderType;
}

struct VersionDirectivePosition
{
	VersionDirectivePosition(int position = 0, int line = -1)
		: position(position)
		, line(line)
	{
	}

	bool hasPosition() const
	{
		return position > 0;
	}

	const int position;
	const int line;
};

static VersionDirectivePosition findVersionDirectivePosition(const char *source)
{
	assert(source);

	// According to the GLSL spec the #version directive must not be
	// preceded by anything but whitespace and comments.
	// In order to not get confused by #version directives within a
	// multiline comment, we need to do some minimal comment parsing
	// while searching for the directive.
	enum {
		Normal,
		StartOfLine,
		PreprocessorDirective,
		CommentStarting,
		MultiLineComment,
		SingleLineComment,
		CommentEnding
	} state = StartOfLine;

	const char *c = source;
	while (*c) {
		switch (state) {
		case PreprocessorDirective:
			if (*c == ' ' || *c == '\t')
				break;
			if (!strncmp(c, "version", strlen("version"))) {
				// Found version directive
				c += strlen("version");
				while (*c && *c != '\n')
					++c;
				int splitPosition = int(c - source + 1);
				int linePosition = int(std::count(source, c, '\n')) + 1;
				return VersionDirectivePosition(splitPosition, linePosition);
			}
			else if (*c == '/')
				state = CommentStarting;
			else if (*c == '\n')
				state = StartOfLine;
			else
				state = Normal;
			break;
		case StartOfLine:
			if (*c == ' ' || *c == '\t')
				break;
			else if (*c == '#') {
				state = PreprocessorDirective;
				break;
			}
			state = Normal;
			// fall through
		case Normal:
			if (*c == '/')
				state = CommentStarting;
			else if (*c == '\n')
				state = StartOfLine;
			break;
		case CommentStarting:
			if (*c == '*')
				state = MultiLineComment;
			else if (*c == '/')
				state = SingleLineComment;
			else
				state = Normal;
			break;
		case MultiLineComment:
			if (*c == '*')
				state = CommentEnding;
			break;
		case SingleLineComment:
			if (*c == '\n')
				state = Normal;
			break;
		case CommentEnding:
			if (*c == '/')
				state = Normal;
			else if (*c != ('*'))
				state = MultiLineComment;
			break;
		}
		++c;
	}

	return VersionDirectivePosition(0, 1);
}

/*!
Sets the \a source code for this shader and compiles it.
Returns \c true if the source was successfully compiled, false otherwise.

\sa compileSourceFile()
*/
bool OpenGLShader::compileSourceCode(const char *source)
{
	// This method breaks the shader code into two parts:
	// 1. Up to and including an optional #version directive.
	// 2. The rest.
	// If a #version directive exists, qualifierDefines and redefineHighp
	// are inserted after. Otherwise they are inserted right at the start.
	// In both cases a #line directive is appended in order to compensate
	// for line number changes in case of compiler errors.

	if (d->shaderGuard && d->shaderGuard->id() && source) {
		const VersionDirectivePosition versionDirectivePosition = findVersionDirectivePosition(source);

		std::vector<const char *> sourceChunks(5);
		std::vector<GLint> sourceChunkLengths(5);

		if (versionDirectivePosition.hasPosition()) {
			// Append source up to and including the #version directive
			sourceChunks.push_back(source);
			sourceChunkLengths.push_back(GLint(versionDirectivePosition.position));
		}


		// Append rest of shader code
		sourceChunks.push_back(source + versionDirectivePosition.position);
		sourceChunkLengths.push_back(GLint(strlen(source + versionDirectivePosition.position)));

		glShaderSource(d->shaderGuard->id(), (GLsizei)sourceChunks.size(), sourceChunks.data(), sourceChunkLengths.data());
		return d->compile(this);
	}
	else {
		return false;
	}
}


/*!
Sets the source code for this shader to the contents of \a fileName
and compiles it.  Returns \c true if the file could be opened and the
source compiled, false otherwise.

\sa compileSourceCode()
*/
std::string file_to_string(const std::string& file_name)
{
	std::ifstream file_stream{ file_name };

	if (file_stream.fail())
	{
		// Error opening file.
		LOG_WARNING << "Error opening file: " << file_name;
	}

	std::ostringstream str_stream{};
	file_stream >> str_stream.rdbuf();  // NOT str_stream << file_stream.rdbuf()

	if (file_stream.fail() && !file_stream.eof())
	{
		// Error reading file.
		LOG_WARNING << "Error reading file: " << file_name;
	}

	return str_stream.str();
}

bool OpenGLShader::compileSourceFile(const std::string& fileName)
{
	std::string contents;
	try {
		contents = file_to_string(fileName);
	}
	catch (...) {
	}

	return compileSourceCode(contents.c_str());
}

/*!
Returns the source code for this shader.

\sa compileSourceCode()
*/
std::string OpenGLShader::sourceCode() const
{
	GLuint shader = d->shaderGuard ? d->shaderGuard->id() : 0;
	if (!shader)
		return "";
	GLint size = 0;
	glGetShaderiv(shader, GL_SHADER_SOURCE_LENGTH, &size);
	if (size <= 0)
		return "";
	GLint len = 0;
	char *source = new char[size];
	glGetShaderSource(shader, size, &len, source);
	std::string src(source);
	delete[] source;
	return src;
}

/*!
Returns \c true if this shader has been compiled; false otherwise.

\sa compileSourceCode(), compileSourceFile()
*/
bool OpenGLShader::isCompiled() const
{
	return d->compiled;
}

/*!
Returns the errors and warnings that occurred during the last compile.

\sa compileSourceCode(), compileSourceFile()
*/
std::string OpenGLShader::log() const
{
	return d->log;
}

/*!
Returns the OpenGL identifier associated with this shader.

\sa OpenGLShaderProgram::programId()
*/
GLuint OpenGLShader::shaderId() const
{
	return d->shaderGuard ? d->shaderGuard->id() : 0;
}

class OpenGLShaderProgramPrivate
{
public:
	OpenGLShaderProgramPrivate()
		: programGuard(0)
		, linked(false)
		, inited(false)
		, removingShaders(false)
	{
	}
	~OpenGLShaderProgramPrivate();

	OpenGLSharedResourceGuard *programGuard;
	bool linked;
	bool inited;
	bool removingShaders;

	std::string log;
	std::vector<OpenGLShader *> shaders;
	std::vector<OpenGLShader *> anonShaders;

	bool hasShader(OpenGLShader::ShaderType type) const;

};

namespace {
	void freeProgramFunc(GLuint id)
	{
		glDeleteProgram(id);
	}
}


OpenGLShaderProgramPrivate::~OpenGLShaderProgramPrivate()
{
	if (programGuard) {
		delete programGuard;
		programGuard = nullptr;
	}
}

bool OpenGLShaderProgramPrivate::hasShader(OpenGLShader::ShaderType type) const
{
	for (OpenGLShader *shader : shaders) {
		if (shader->shaderType() == type)
			return true;
	}
	return false;
}

/*!
Constructs a new shader program and attaches it to \a parent.
The program will be invalid until addShader() is called.

The shader program will be associated with the current QOpenGLContext.

\sa addShader()
*/
OpenGLShaderProgram::OpenGLShaderProgram()
{
	d = new OpenGLShaderProgramPrivate();
}

/*!
Deletes this shader program.
*/
OpenGLShaderProgram::~OpenGLShaderProgram()
{
	delete d;
}

/*!
Requests the shader program's id to be created immediately. Returns \c true
if successful; \c false otherwise.

This function is primarily useful when combining OpenGLShaderProgram
with other OpenGL functions that operate directly on the shader
program id, like \c {GL_OES_get_program_binary}.

When the shader program is used normally, the shader program's id will
be created on demand.

\sa programId()

\since 5.3
*/
bool OpenGLShaderProgram::create()
{
	return init();
}

bool OpenGLShaderProgram::init()
{
	if ((d->programGuard && d->programGuard->id()) || d->inited)
		return true;
	d->inited = true;

	GLuint program = glCreateProgram();
	if (!program) {
		LOG_WARNING << "OpenGLShaderProgram: could not create shader program";
		return false;
	}
	if (d->programGuard)
		delete d->programGuard;
	d->programGuard = new OpenGLSharedResourceGuard(program, freeProgramFunc);
	return true;
}

template<class C, class T>
auto contains(const C& v, const T& x)
-> decltype(end(v), true)
{
	return end(v) != std::find(begin(v), end(v), x);
}

template<class C, class T>
auto removeAll(C& v, const T& x)
{
	return v.erase(std::remove(v.begin(), v.end(), x), v.end());
}

template<class C>
auto deleteAll(C& v)
{
	for (auto entry : v)
	{
		delete entry;
	}
	v.clear();
}

/*!
Adds a compiled \a shader to this shader program.  Returns \c true
if the shader could be added, or false otherwise.

Ownership of the \a shader object remains with the caller.
It will not be deleted when this OpenGLShaderProgram instance
is deleted.  This allows the caller to add the same shader
to multiple shader programs.

\sa addShaderFromSourceCode(), addShaderFromSourceFile()
\sa removeShader(), link(), removeAllShaders()
*/
bool OpenGLShaderProgram::addShader(OpenGLShader *shader)
{
	if (!init())
		return false;
	if (contains(d->shaders, shader))
		return true;    // Already added to this shader program.
	if (d->programGuard && d->programGuard->id() && shader) {
		if (!shader->d->shaderGuard || !shader->d->shaderGuard->id())
			return false;
		//if (d->programGuard->group() != shader->d->shaderGuard->group()) {
		//	LOG_WARNING << "OpenGLShaderProgram::addShader: Program and shader are not associated with same context.";
		//	return false;
		//}
		glAttachShader(d->programGuard->id(), shader->d->shaderGuard->id());
		d->linked = false;  // Program needs to be relinked.
		d->shaders.push_back(shader);
		return true;
	}
	else {
		return false;
	}
}

/*!
Compiles \a source as a shader of the specified \a type and
adds it to this shader program.  Returns \c true if compilation
was successful, false otherwise.  The compilation errors
and warnings will be made available via log().

This function is intended to be a short-cut for quickly
adding vertex and fragment shaders to a shader program without
creating an instance of OpenGLShader first.

\sa addShader(), addShaderFromSourceFile()
\sa removeShader(), link(), log(), removeAllShaders()
*/
bool OpenGLShaderProgram::addShaderFromSourceCode(OpenGLShader::ShaderType type, const char *source)
{
	if (!init())
		return false;
	OpenGLShader *shader = new OpenGLShader(type);
	if (!shader->compileSourceCode(source)) {
		d->log = shader->log();
		delete shader;
		return false;
	}
	d->anonShaders.push_back(shader);
	return addShader(shader);
}

/*!
Compiles the contents of \a fileName as a shader of the specified
\a type and adds it to this shader program.  Returns \c true if
compilation was successful, false otherwise.  The compilation errors
and warnings will be made available via log().

This function is intended to be a short-cut for quickly
adding vertex and fragment shaders to a shader program without
creating an instance of OpenGLShader first.

\sa addShader(), addShaderFromSourceCode()
*/
bool OpenGLShaderProgram::addShaderFromSourceFile
(OpenGLShader::ShaderType type, const std::string& fileName)
{
	if (!init())
		return false;
	OpenGLShader *shader = new OpenGLShader(type);
	if (!shader->compileSourceFile(fileName)) {
		d->log = shader->log();
		delete shader;
		return false;
	}
	d->anonShaders.push_back(shader);
	return addShader(shader);
}


/*!
Removes \a shader from this shader program.  The object is not deleted.

The shader program must be valid in the current QOpenGLContext.

\sa addShader(), link(), removeAllShaders()
*/
void OpenGLShaderProgram::removeShader(OpenGLShader *shader)
{
	if (d->programGuard && d->programGuard->id()
		&& shader && shader->d->shaderGuard)
	{
		glDetachShader(d->programGuard->id(), shader->d->shaderGuard->id());
	}
	d->linked = false;  // Program needs to be relinked.
	if (shader) {
		removeAll(d->shaders, shader);
		removeAll(d->anonShaders, shader);
	}
}

/*!
Returns a list of all shaders that have been added to this shader
program using addShader().

\sa addShader(), removeShader()
*/
std::vector<OpenGLShader *> OpenGLShaderProgram::shaders() const
{
	return d->shaders;
}

/*!
Removes all of the shaders that were added to this program previously.
The OpenGLShader objects for the shaders will not be deleted if they
were constructed externally.  OpenGLShader objects that are constructed
internally by OpenGLShaderProgram will be deleted.

\sa addShader(), removeShader()
*/
void OpenGLShaderProgram::removeAllShaders()
{
	d->removingShaders = true;
	for (OpenGLShader *shader : (d->shaders)) {
		if (d->programGuard && d->programGuard->id()
			&& shader && shader->d->shaderGuard)
		{
			glDetachShader(d->programGuard->id(), shader->d->shaderGuard->id());
		}
	}
	// Delete shader objects that were created anonymously.
	deleteAll(d->anonShaders);
	d->shaders.clear();
	d->anonShaders.clear();
	d->linked = false;  // Program needs to be relinked.
	d->removingShaders = false;
}

/*!
Links together the shaders that were added to this program with
addShader().  Returns \c true if the link was successful or
false otherwise.  If the link failed, the error messages can
be retrieved with log().

Subclasses can override this function to initialize attributes
and uniform variables for use in specific shader programs.

If the shader program was already linked, calling this
function again will force it to be re-linked.

When shaders were added to this program via
addCacheableShaderFromSourceCode() or addCacheableShaderFromSourceFile(),
program binaries are supported, and a cached binary is available on disk,
actual compilation and linking are skipped. Instead, link() will initialize
the program with the binary blob via glProgramBinary(). If there is no
cached version of the program or it was generated with a different driver
version, the shaders will be compiled from source and the program will get
linked normally. This allows seamless upgrading of the graphics drivers,
without having to worry about potentially incompatible binary formats.

\sa addShader(), log()
*/
bool OpenGLShaderProgram::link()
{
	GLuint program = d->programGuard ? d->programGuard->id() : 0;
	if (!program)
		return false;

	GLint value;
	if (d->shaders.empty()) {
		// If there are no explicit shaders, then it is possible that the
		// application added a program binary with glProgramBinaryOES(), or
		// otherwise populated the shaders itself. This is also the case when
		// we are recursively called back from linkBinary() after a successful
		// glProgramBinary(). Check to see if the program is already linked and
		// bail out if so.
		value = 0;
		glGetProgramiv(program, GL_LINK_STATUS, &value);
		d->linked = (value != 0);
		if (d->linked)
			return true;
	}

	glLinkProgram(program);
	value = 0;
	glGetProgramiv(program, GL_LINK_STATUS, &value);
	d->linked = (value != 0);
	value = 0;
	glGetProgramiv(program, GL_INFO_LOG_LENGTH, &value);
	d->log = std::string();
	if (value > 1) {
		char *logbuf = new char[value];
		GLint len;
		glGetProgramInfoLog(program, value, &len, logbuf);
		d->log = std::string(logbuf);
		if (!d->linked) {
			LOG_WARNING<<"OpenGLShader::link: " << d->log;
		}
		delete[] logbuf;
	}
	return d->linked;
}

/*!
Returns \c true if this shader program has been linked; false otherwise.

\sa link()
*/
bool OpenGLShaderProgram::isLinked() const
{
	return d->linked;
}

/*!
Returns the errors and warnings that occurred during the last link()
or addShader() with explicitly specified source code.

\sa link()
*/
std::string OpenGLShaderProgram::log() const
{
	return d->log;
}

/*!
Binds this shader program to the active QOpenGLContext and makes
it the current shader program.  Any previously bound shader program
is released.  This is equivalent to calling \c{glUseProgram()} on
programId().  Returns \c true if the program was successfully bound;
false otherwise.  If the shader program has not yet been linked,
or it needs to be re-linked, this function will call link().

\sa link(), release()
*/
bool OpenGLShaderProgram::bind()
{
	GLuint program = d->programGuard ? d->programGuard->id() : 0;
	if (!program)
		return false;
	if (!d->linked && !link())
		return false;

	glUseProgram(program);
	return true;
}

/*!
Releases the active shader program from the current QOpenGLContext.
This is equivalent to calling \c{glUseProgram(0)}.

\sa bind()
*/
void OpenGLShaderProgram::release()
{
	glUseProgram(0);
}

/*!
Returns the OpenGL identifier associated with this shader program.

\sa OpenGLShader::shaderId()
*/
GLuint OpenGLShaderProgram::programId() const
{
	GLuint id = d->programGuard ? d->programGuard->id() : 0;
	if (id)
		return id;

	// Create the identifier if we don't have one yet.  This is for
	// applications that want to create the attached shader configuration
	// themselves, particularly those using program binaries.
	if (!const_cast<OpenGLShaderProgram *>(this)->init())
		return 0;
	return d->programGuard ? d->programGuard->id() : 0;
}

/*!
Binds the attribute \a name to the specified \a location.  This
function can be called before or after the program has been linked.
Any attributes that have not been explicitly bound when the program
is linked will be assigned locations automatically.

When this function is called after the program has been linked,
the program will need to be relinked for the change to take effect.

\sa attributeLocation()
*/
void OpenGLShaderProgram::bindAttributeLocation(const char *name, int location)
{
	if (!init() || !d->programGuard || !d->programGuard->id())
		return;
	glBindAttribLocation(d->programGuard->id(), location, name);
	d->linked = false;  // Program needs to be relinked.
}


/*!
Returns the location of the attribute \a name within this shader
program's parameter list.  Returns -1 if \a name is not a valid
attribute for this shader program.

\sa uniformLocation(), bindAttributeLocation()
*/
int OpenGLShaderProgram::attributeLocation(const char *name) const
{
	if (d->linked && d->programGuard && d->programGuard->id()) {
		return glGetAttribLocation(d->programGuard->id(), name);
	}
	else {
		LOG_WARNING<<"OpenGLShaderProgram::attributeLocation(" << name << "): shader program is not linked";
		return -1;
	}
}

/*!
Sets the attribute at \a location in the current context to \a value.

\sa setUniformValue()
*/
void OpenGLShaderProgram::setAttributeValue(int location, GLfloat value)
{
	if (location != -1)
		glVertexAttrib1fv(location, &value);
}

/*!
\overload

Sets the attribute called \a name in the current context to \a value.

\sa setUniformValue()
*/
void OpenGLShaderProgram::setAttributeValue(const char *name, GLfloat value)
{
	setAttributeValue(attributeLocation(name), value);
}

/*!
Sets the attribute at \a location in the current context to
the 2D vector (\a x, \a y).

\sa setUniformValue()
*/
void OpenGLShaderProgram::setAttributeValue(int location, GLfloat x, GLfloat y)
{
	if (location != -1) {
		GLfloat values[2] = { x, y };
		glVertexAttrib2fv(location, values);
	}
}

/*!
\overload

Sets the attribute called \a name in the current context to
the 2D vector (\a x, \a y).

\sa setUniformValue()
*/
void OpenGLShaderProgram::setAttributeValue(const char *name, GLfloat x, GLfloat y)
{
	setAttributeValue(attributeLocation(name), x, y);
}

/*!
Sets the attribute at \a location in the current context to
the 3D vector (\a x, \a y, \a z).

\sa setUniformValue()
*/
void OpenGLShaderProgram::setAttributeValue
(int location, GLfloat x, GLfloat y, GLfloat z)
{
	if (location != -1) {
		GLfloat values[3] = { x, y, z };
		glVertexAttrib3fv(location, values);
	}
}

/*!
\overload

Sets the attribute called \a name in the current context to
the 3D vector (\a x, \a y, \a z).

\sa setUniformValue()
*/
void OpenGLShaderProgram::setAttributeValue
(const char *name, GLfloat x, GLfloat y, GLfloat z)
{
	setAttributeValue(attributeLocation(name), x, y, z);
}

/*!
Sets the attribute at \a location in the current context to
the 4D vector (\a x, \a y, \a z, \a w).

\sa setUniformValue()
*/
void OpenGLShaderProgram::setAttributeValue
(int location, GLfloat x, GLfloat y, GLfloat z, GLfloat w)
{
	if (location != -1) {
		GLfloat values[4] = { x, y, z, w };
		glVertexAttrib4fv(location, values);
	}
}

/*!
\overload

Sets the attribute called \a name in the current context to
the 4D vector (\a x, \a y, \a z, \a w).

\sa setUniformValue()
*/
void OpenGLShaderProgram::setAttributeValue
(const char *name, GLfloat x, GLfloat y, GLfloat z, GLfloat w)
{
	setAttributeValue(attributeLocation(name), x, y, z, w);
}

/*!
Sets the attribute at \a location in the current context to \a value.

\sa setUniformValue()
*/
void OpenGLShaderProgram::setAttributeValue(int location, const glm::vec2& value)
{
	if (location != -1)
		glVertexAttrib2fv(location, reinterpret_cast<const GLfloat *>(&value));
}

/*!
\overload

Sets the attribute called \a name in the current context to \a value.

\sa setUniformValue()
*/
void OpenGLShaderProgram::setAttributeValue(const char *name, const glm::vec2& value)
{
	setAttributeValue(attributeLocation(name), value);
}

/*!
Sets the attribute at \a location in the current context to \a value.

\sa setUniformValue()
*/
void OpenGLShaderProgram::setAttributeValue(int location, const glm::vec3& value)
{
	if (location != -1)
		glVertexAttrib3fv(location, reinterpret_cast<const GLfloat *>(&value));
}

/*!
\overload

Sets the attribute called \a name in the current context to \a value.

\sa setUniformValue()
*/
void OpenGLShaderProgram::setAttributeValue(const char *name, const glm::vec3& value)
{
	setAttributeValue(attributeLocation(name), value);
}

/*!
Sets the attribute at \a location in the current context to \a value.

\sa setUniformValue()
*/
void OpenGLShaderProgram::setAttributeValue(int location, const glm::vec4& value)
{
	if (location != -1)
		glVertexAttrib4fv(location, reinterpret_cast<const GLfloat *>(&value));
}

/*!
\overload

Sets the attribute called \a name in the current context to \a value.

\sa setUniformValue()
*/
void OpenGLShaderProgram::setAttributeValue(const char *name, const glm::vec4& value)
{
	setAttributeValue(attributeLocation(name), value);
}

/*!
Sets the attribute at \a location in the current context to the
contents of \a values, which contains \a columns elements, each
consisting of \a rows elements.  The \a rows value should be
1, 2, 3, or 4.  This function is typically used to set matrix
values and column vectors.

\sa setUniformValue()
*/
void OpenGLShaderProgram::setAttributeValue
(int location, const GLfloat *values, int columns, int rows)
{
	if (rows < 1 || rows > 4) {
		LOG_WARNING<<"OpenGLShaderProgram::setAttributeValue: rows " << rows << " not supported";
		return;
	}
	if (location != -1) {
		while (columns-- > 0) {
			if (rows == 1)
				glVertexAttrib1fv(location, values);
			else if (rows == 2)
				glVertexAttrib2fv(location, values);
			else if (rows == 3)
				glVertexAttrib3fv(location, values);
			else
				glVertexAttrib4fv(location, values);
			values += rows;
			++location;
		}
	}
}

/*!
\overload

Sets the attribute called \a name in the current context to the
contents of \a values, which contains \a columns elements, each
consisting of \a rows elements.  The \a rows value should be
1, 2, 3, or 4.  This function is typically used to set matrix
values and column vectors.

\sa setUniformValue()
*/
void OpenGLShaderProgram::setAttributeValue
(const char *name, const GLfloat *values, int columns, int rows)
{
	setAttributeValue(attributeLocation(name), values, columns, rows);
}

/*!
Sets an array of vertex \a values on the attribute at \a location
in this shader program.  The \a tupleSize indicates the number of
components per vertex (1, 2, 3, or 4), and the \a stride indicates
the number of bytes between vertices.  A default \a stride value
of zero indicates that the vertices are densely packed in \a values.

The array will become active when enableAttributeArray() is called
on the \a location.  Otherwise the value specified with
setAttributeValue() for \a location will be used.

\sa setAttributeValue(), setUniformValue(), enableAttributeArray()
\sa disableAttributeArray()
*/
void OpenGLShaderProgram::setAttributeArray
(int location, const GLfloat *values, int tupleSize, int stride)
{
	if (location != -1) {
		glVertexAttribPointer(location, tupleSize, GL_FLOAT, GL_FALSE,
			stride, values);
	}
}

/*!
Sets an array of 2D vertex \a values on the attribute at \a location
in this shader program.  The \a stride indicates the number of bytes
between vertices.  A default \a stride value of zero indicates that
the vertices are densely packed in \a values.

The array will become active when enableAttributeArray() is called
on the \a location.  Otherwise the value specified with
setAttributeValue() for \a location will be used.

\sa setAttributeValue(), setUniformValue(), enableAttributeArray()
\sa disableAttributeArray()
*/
void OpenGLShaderProgram::setAttributeArray
(int location, const glm::vec2 *values, int stride)
{
	if (location != -1) {
		glVertexAttribPointer(location, 2, GL_FLOAT, GL_FALSE,
			stride, values);
	}
}

/*!
Sets an array of 3D vertex \a values on the attribute at \a location
in this shader program.  The \a stride indicates the number of bytes
between vertices.  A default \a stride value of zero indicates that
the vertices are densely packed in \a values.

The array will become active when enableAttributeArray() is called
on the \a location.  Otherwise the value specified with
setAttributeValue() for \a location will be used.

\sa setAttributeValue(), setUniformValue(), enableAttributeArray()
\sa disableAttributeArray()
*/
void OpenGLShaderProgram::setAttributeArray
(int location, const glm::vec3 *values, int stride)
{
	if (location != -1) {
		glVertexAttribPointer(location, 3, GL_FLOAT, GL_FALSE,
			stride, values);
	}
}

/*!
Sets an array of 4D vertex \a values on the attribute at \a location
in this shader program.  The \a stride indicates the number of bytes
between vertices.  A default \a stride value of zero indicates that
the vertices are densely packed in \a values.

The array will become active when enableAttributeArray() is called
on the \a location.  Otherwise the value specified with
setAttributeValue() for \a location will be used.

\sa setAttributeValue(), setUniformValue(), enableAttributeArray()
\sa disableAttributeArray()
*/
void OpenGLShaderProgram::setAttributeArray
(int location, const glm::vec4 *values, int stride)
{
	if (location != -1) {
		glVertexAttribPointer(location, 4, GL_FLOAT, GL_FALSE,
			stride, values);
	}
}

/*!
Sets an array of vertex \a values on the attribute at \a location
in this shader program.  The \a stride indicates the number of bytes
between vertices.  A default \a stride value of zero indicates that
the vertices are densely packed in \a values.

The \a type indicates the type of elements in the \a values array,
usually \c{GL_FLOAT}, \c{GL_UNSIGNED_BYTE}, etc.  The \a tupleSize
indicates the number of components per vertex: 1, 2, 3, or 4.

The array will become active when enableAttributeArray() is called
on the \a location.  Otherwise the value specified with
setAttributeValue() for \a location will be used.

The setAttributeBuffer() function can be used to set the attribute
array to an offset within a vertex buffer.

\note Normalization will be enabled. If this is not desired, call
glVertexAttribPointer directly through QOpenGLFunctions.

\sa setAttributeValue(), setUniformValue(), enableAttributeArray()
\sa disableAttributeArray(), setAttributeBuffer()
*/
void OpenGLShaderProgram::setAttributeArray
(int location, GLenum type, const void *values, int tupleSize, int stride)
{
	if (location != -1) {
		glVertexAttribPointer(location, tupleSize, type, GL_TRUE,
			stride, values);
	}
}

/*!
\overload

Sets an array of vertex \a values on the attribute called \a name
in this shader program.  The \a tupleSize indicates the number of
components per vertex (1, 2, 3, or 4), and the \a stride indicates
the number of bytes between vertices.  A default \a stride value
of zero indicates that the vertices are densely packed in \a values.

The array will become active when enableAttributeArray() is called
on \a name.  Otherwise the value specified with setAttributeValue()
for \a name will be used.

\sa setAttributeValue(), setUniformValue(), enableAttributeArray()
\sa disableAttributeArray()
*/
void OpenGLShaderProgram::setAttributeArray
(const char *name, const GLfloat *values, int tupleSize, int stride)
{
	setAttributeArray(attributeLocation(name), values, tupleSize, stride);
}

/*!
\overload

Sets an array of 2D vertex \a values on the attribute called \a name
in this shader program.  The \a stride indicates the number of bytes
between vertices.  A default \a stride value of zero indicates that
the vertices are densely packed in \a values.

The array will become active when enableAttributeArray() is called
on \a name.  Otherwise the value specified with setAttributeValue()
for \a name will be used.

\sa setAttributeValue(), setUniformValue(), enableAttributeArray()
\sa disableAttributeArray()
*/
void OpenGLShaderProgram::setAttributeArray
(const char *name, const glm::vec2 *values, int stride)
{
	setAttributeArray(attributeLocation(name), values, stride);
}

/*!
\overload

Sets an array of 3D vertex \a values on the attribute called \a name
in this shader program.  The \a stride indicates the number of bytes
between vertices.  A default \a stride value of zero indicates that
the vertices are densely packed in \a values.

The array will become active when enableAttributeArray() is called
on \a name.  Otherwise the value specified with setAttributeValue()
for \a name will be used.

\sa setAttributeValue(), setUniformValue(), enableAttributeArray()
\sa disableAttributeArray()
*/
void OpenGLShaderProgram::setAttributeArray
(const char *name, const glm::vec3 *values, int stride)
{
	setAttributeArray(attributeLocation(name), values, stride);
}

/*!
\overload

Sets an array of 4D vertex \a values on the attribute called \a name
in this shader program.  The \a stride indicates the number of bytes
between vertices.  A default \a stride value of zero indicates that
the vertices are densely packed in \a values.

The array will become active when enableAttributeArray() is called
on \a name.  Otherwise the value specified with setAttributeValue()
for \a name will be used.

\sa setAttributeValue(), setUniformValue(), enableAttributeArray()
\sa disableAttributeArray()
*/
void OpenGLShaderProgram::setAttributeArray
(const char *name, const glm::vec4 *values, int stride)
{
	setAttributeArray(attributeLocation(name), values, stride);
}

/*!
\overload

Sets an array of vertex \a values on the attribute called \a name
in this shader program.  The \a stride indicates the number of bytes
between vertices.  A default \a stride value of zero indicates that
the vertices are densely packed in \a values.

The \a type indicates the type of elements in the \a values array,
usually \c{GL_FLOAT}, \c{GL_UNSIGNED_BYTE}, etc.  The \a tupleSize
indicates the number of components per vertex: 1, 2, 3, or 4.

The array will become active when enableAttributeArray() is called
on the \a name.  Otherwise the value specified with
setAttributeValue() for \a name will be used.

The setAttributeBuffer() function can be used to set the attribute
array to an offset within a vertex buffer.

\sa setAttributeValue(), setUniformValue(), enableAttributeArray()
\sa disableAttributeArray(), setAttributeBuffer()
*/
void OpenGLShaderProgram::setAttributeArray
(const char *name, GLenum type, const void *values, int tupleSize, int stride)
{
	setAttributeArray(attributeLocation(name), type, values, tupleSize, stride);
}

/*!
Sets an array of vertex values on the attribute at \a location in
this shader program, starting at a specific \a offset in the
currently bound vertex buffer.  The \a stride indicates the number
of bytes between vertices.  A default \a stride value of zero
indicates that the vertices are densely packed in the value array.

The \a type indicates the type of elements in the vertex value
array, usually \c{GL_FLOAT}, \c{GL_UNSIGNED_BYTE}, etc.  The \a
tupleSize indicates the number of components per vertex: 1, 2, 3,
or 4.

The array will become active when enableAttributeArray() is called
on the \a location.  Otherwise the value specified with
setAttributeValue() for \a location will be used.

\note Normalization will be enabled. If this is not desired, call
glVertexAttribPointer directly through QOpenGLFunctions.

\sa setAttributeArray()
*/
void OpenGLShaderProgram::setAttributeBuffer
(int location, GLenum type, int offset, int tupleSize, int stride)
{
	if (location != -1) {
		glVertexAttribPointer(location, tupleSize, type, GL_TRUE, stride,
			reinterpret_cast<const void *>(intptr_t(offset)));
	}
}

/*!
\overload

Sets an array of vertex values on the attribute called \a name
in this shader program, starting at a specific \a offset in the
currently bound vertex buffer.  The \a stride indicates the number
of bytes between vertices.  A default \a stride value of zero
indicates that the vertices are densely packed in the value array.

The \a type indicates the type of elements in the vertex value
array, usually \c{GL_FLOAT}, \c{GL_UNSIGNED_BYTE}, etc.  The \a
tupleSize indicates the number of components per vertex: 1, 2, 3,
or 4.

The array will become active when enableAttributeArray() is called
on the \a name.  Otherwise the value specified with
setAttributeValue() for \a name will be used.

\sa setAttributeArray()
*/
void OpenGLShaderProgram::setAttributeBuffer
(const char *name, GLenum type, int offset, int tupleSize, int stride)
{
	setAttributeBuffer(attributeLocation(name), type, offset, tupleSize, stride);
}

/*!
Enables the vertex array at \a location in this shader program
so that the value set by setAttributeArray() on \a location
will be used by the shader program.

\sa disableAttributeArray(), setAttributeArray(), setAttributeValue()
\sa setUniformValue()
*/
void OpenGLShaderProgram::enableAttributeArray(int location)
{
	if (location != -1)
		glEnableVertexAttribArray(location);
}

/*!
\overload

Enables the vertex array called \a name in this shader program
so that the value set by setAttributeArray() on \a name
will be used by the shader program.

\sa disableAttributeArray(), setAttributeArray(), setAttributeValue()
\sa setUniformValue()
*/
void OpenGLShaderProgram::enableAttributeArray(const char *name)
{
	enableAttributeArray(attributeLocation(name));
}

/*!
Disables the vertex array at \a location in this shader program
that was enabled by a previous call to enableAttributeArray().

\sa enableAttributeArray(), setAttributeArray(), setAttributeValue()
\sa setUniformValue()
*/
void OpenGLShaderProgram::disableAttributeArray(int location)
{
	if (location != -1)
		glDisableVertexAttribArray(location);
}

/*!
\overload

Disables the vertex array called \a name in this shader program
that was enabled by a previous call to enableAttributeArray().

\sa enableAttributeArray(), setAttributeArray(), setAttributeValue()
\sa setUniformValue()
*/
void OpenGLShaderProgram::disableAttributeArray(const char *name)
{
	disableAttributeArray(attributeLocation(name));
}

/*!
Returns the location of the uniform variable \a name within this shader
program's parameter list.  Returns -1 if \a name is not a valid
uniform variable for this shader program.

\sa attributeLocation()
*/
int OpenGLShaderProgram::uniformLocation(const char *name) const
{
	if (d->linked && d->programGuard && d->programGuard->id()) {
		int iLoc = glGetUniformLocation(d->programGuard->id(), name);
		if (iLoc == -1) {
			LOG_WARNING << "Failed to find " << name << " uniform";
			std::cerr << "GLBasicVolumeShader: Failed to bind texture uniform" << std::endl;
		}
		return iLoc;
	}
	else {
		LOG_WARNING<<"OpenGLShaderProgram::uniformLocation(" << name << "): shader program is not linked";
		return -1;
	}
}

/*!
Sets the uniform variable at \a location in the current context to \a value.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue(int location, GLfloat value)
{
	if (location != -1)
		glUniform1fv(location, 1, &value);
}

/*!
\overload

Sets the uniform variable called \a name in the current context
to \a value.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue(const char *name, GLfloat value)
{
	setUniformValue(uniformLocation(name), value);
}

/*!
Sets the uniform variable at \a location in the current context to \a value.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue(int location, GLint value)
{
	if (location != -1)
		glUniform1i(location, value);
}

/*!
\overload

Sets the uniform variable called \a name in the current context
to \a value.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue(const char *name, GLint value)
{
	setUniformValue(uniformLocation(name), value);
}

/*!
Sets the uniform variable at \a location in the current context to \a value.
This function should be used when setting sampler values.

\note This function is not aware of unsigned int support in modern OpenGL
versions and therefore treats \a value as a GLint and calls glUniform1i.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue(int location, GLuint value)
{
	if (location != -1)
		glUniform1i(location, value);
}

/*!
\overload

Sets the uniform variable called \a name in the current context
to \a value.  This function should be used when setting sampler values.

\note This function is not aware of unsigned int support in modern OpenGL
versions and therefore treats \a value as a GLint and calls glUniform1i.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue(const char *name, GLuint value)
{
	setUniformValue(uniformLocation(name), value);
}

/*!
Sets the uniform variable at \a location in the current context to
the 2D vector (\a x, \a y).

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue(int location, GLfloat x, GLfloat y)
{
	if (location != -1) {
		GLfloat values[2] = { x, y };
		glUniform2fv(location, 1, values);
	}
}

/*!
\overload

Sets the uniform variable called \a name in the current context to
the 2D vector (\a x, \a y).

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue(const char *name, GLfloat x, GLfloat y)
{
	setUniformValue(uniformLocation(name), x, y);
}

/*!
Sets the uniform variable at \a location in the current context to
the 3D vector (\a x, \a y, \a z).

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue
(int location, GLfloat x, GLfloat y, GLfloat z)
{
	if (location != -1) {
		GLfloat values[3] = { x, y, z };
		glUniform3fv(location, 1, values);
	}
}

/*!
\overload

Sets the uniform variable called \a name in the current context to
the 3D vector (\a x, \a y, \a z).

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue
(const char *name, GLfloat x, GLfloat y, GLfloat z)
{
	setUniformValue(uniformLocation(name), x, y, z);
}

/*!
Sets the uniform variable at \a location in the current context to
the 4D vector (\a x, \a y, \a z, \a w).

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue
(int location, GLfloat x, GLfloat y, GLfloat z, GLfloat w)
{
	if (location != -1) {
		GLfloat values[4] = { x, y, z, w };
		glUniform4fv(location, 1, values);
	}
}

/*!
\overload

Sets the uniform variable called \a name in the current context to
the 4D vector (\a x, \a y, \a z, \a w).

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue
(const char *name, GLfloat x, GLfloat y, GLfloat z, GLfloat w)
{
	setUniformValue(uniformLocation(name), x, y, z, w);
}

/*!
Sets the uniform variable at \a location in the current context to \a value.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue(int location, const glm::vec2& value)
{
	if (location != -1)
		glUniform2fv(location, 1, reinterpret_cast<const GLfloat *>(&value));
}

/*!
\overload

Sets the uniform variable called \a name in the current context
to \a value.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue(const char *name, const glm::vec2& value)
{
	setUniformValue(uniformLocation(name), value);
}

/*!
Sets the uniform variable at \a location in the current context to \a value.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue(int location, const glm::vec3& value)
{
	if (location != -1)
		glUniform3fv(location, 1, reinterpret_cast<const GLfloat *>(&value));
}

/*!
\overload

Sets the uniform variable called \a name in the current context
to \a value.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue(const char *name, const glm::vec3& value)
{
	setUniformValue(uniformLocation(name), value);
}

/*!
Sets the uniform variable at \a location in the current context to \a value.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue(int location, const glm::vec4& value)
{
	if (location != -1)
		glUniform4fv(location, 1, reinterpret_cast<const GLfloat *>(&value));
}

/*!
\overload

Sets the uniform variable called \a name in the current context
to \a value.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue(const char *name, const glm::vec4& value)
{
	setUniformValue(uniformLocation(name), value);
}

/*!
Sets the uniform variable at \a location in the current context
to a 2x2 matrix \a value.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue(int location, const glm::mat2& value)
{
	glUniformMatrix2fv(location, 1, GL_FALSE, glm::value_ptr(value));
}

/*!
\overload

Sets the uniform variable called \a name in the current context
to a 2x2 matrix \a value.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue(const char *name, const glm::mat2& value)
{
	setUniformValue(uniformLocation(name), value);
}

/*!
Sets the uniform variable at \a location in the current context
to a 3x3 matrix \a value.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue(int location, const glm::mat3& value)
{
	glUniformMatrix3fv(location, 1, GL_FALSE, glm::value_ptr(value));
}

/*!
\overload

Sets the uniform variable called \a name in the current context
to a 3x3 matrix \a value.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue(const char *name, const glm::mat3& value)
{
	setUniformValue(uniformLocation(name), value);
}

/*!
Sets the uniform variable at \a location in the current context
to a 4x4 matrix \a value.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue(int location, const glm::mat4& value)
{
	glUniformMatrix4fv(location, 1, GL_FALSE, glm::value_ptr(value));
}

/*!
\overload

Sets the uniform variable called \a name in the current context
to a 4x4 matrix \a value.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue(const char *name, const glm::mat4& value)
{
	setUniformValue(uniformLocation(name), value);
}

/*!
\overload

Sets the uniform variable at \a location in the current context
to a 2x2 matrix \a value.  The matrix elements must be specified
in column-major order.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue(int location, const GLfloat value[2][2])
{
	if (location != -1)
		glUniformMatrix2fv(location, 1, GL_FALSE, value[0]);
}

/*!
\overload

Sets the uniform variable at \a location in the current context
to a 3x3 matrix \a value.  The matrix elements must be specified
in column-major order.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue(int location, const GLfloat value[3][3])
{
	if (location != -1)
		glUniformMatrix3fv(location, 1, GL_FALSE, value[0]);
}

/*!
\overload

Sets the uniform variable at \a location in the current context
to a 4x4 matrix \a value.  The matrix elements must be specified
in column-major order.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue(int location, const GLfloat value[4][4])
{
	if (location != -1)
		glUniformMatrix4fv(location, 1, GL_FALSE, value[0]);
}


/*!
\overload

Sets the uniform variable called \a name in the current context
to a 2x2 matrix \a value.  The matrix elements must be specified
in column-major order.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue(const char *name, const GLfloat value[2][2])
{
	setUniformValue(uniformLocation(name), value);
}

/*!
\overload

Sets the uniform variable called \a name in the current context
to a 3x3 matrix \a value.  The matrix elements must be specified
in column-major order.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue(const char *name, const GLfloat value[3][3])
{
	setUniformValue(uniformLocation(name), value);
}

/*!
\overload

Sets the uniform variable called \a name in the current context
to a 4x4 matrix \a value.  The matrix elements must be specified
in column-major order.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValue(const char *name, const GLfloat value[4][4])
{
	setUniformValue(uniformLocation(name), value);
}


/*!
Sets the uniform variable array at \a location in the current
context to the \a count elements of \a values.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValueArray(int location, const GLint *values, int count)
{
	if (location != -1)
		glUniform1iv(location, count, values);
}

/*!
\overload

Sets the uniform variable array called \a name in the current
context to the \a count elements of \a values.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValueArray
(const char *name, const GLint *values, int count)
{
	setUniformValueArray(uniformLocation(name), values, count);
}

/*!
Sets the uniform variable array at \a location in the current
context to the \a count elements of \a values.  This overload
should be used when setting an array of sampler values.

\note This function is not aware of unsigned int support in modern OpenGL
versions and therefore treats \a values as a GLint and calls glUniform1iv.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValueArray(int location, const GLuint *values, int count)
{
	if (location != -1)
		glUniform1iv(location, count, reinterpret_cast<const GLint *>(values));
}

/*!
\overload

Sets the uniform variable array called \a name in the current
context to the \a count elements of \a values.  This overload
should be used when setting an array of sampler values.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValueArray
(const char *name, const GLuint *values, int count)
{
	setUniformValueArray(uniformLocation(name), values, count);
}

/*!
Sets the uniform variable array at \a location in the current
context to the \a count elements of \a values.  Each element
has \a tupleSize components.  The \a tupleSize must be 1, 2, 3, or 4.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValueArray(int location, const GLfloat *values, int count, int tupleSize)
{
	if (location != -1) {
		if (tupleSize == 1)
			glUniform1fv(location, count, values);
		else if (tupleSize == 2)
			glUniform2fv(location, count, values);
		else if (tupleSize == 3)
			glUniform3fv(location, count, values);
		else if (tupleSize == 4)
			glUniform4fv(location, count, values);
		else
			LOG_WARNING<<"OpenGLShaderProgram::setUniformValue: size " << tupleSize << " not supported";
	}
}

/*!
\overload

Sets the uniform variable array called \a name in the current
context to the \a count elements of \a values.  Each element
has \a tupleSize components.  The \a tupleSize must be 1, 2, 3, or 4.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValueArray
(const char *name, const GLfloat *values, int count, int tupleSize)
{
	setUniformValueArray(uniformLocation(name), values, count, tupleSize);
}

/*!
Sets the uniform variable array at \a location in the current
context to the \a count 2D vector elements of \a values.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValueArray(int location, const glm::vec2 *values, int count)
{
	if (location != -1)
		glUniform2fv(location, count, reinterpret_cast<const GLfloat *>(values));
}

/*!
\overload

Sets the uniform variable array called \a name in the current
context to the \a count 2D vector elements of \a values.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValueArray(const char *name, const glm::vec2 *values, int count)
{
	setUniformValueArray(uniformLocation(name), values, count);
}

/*!
Sets the uniform variable array at \a location in the current
context to the \a count 3D vector elements of \a values.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValueArray(int location, const glm::vec3 *values, int count)
{
	if (location != -1)
		glUniform3fv(location, count, reinterpret_cast<const GLfloat *>(values));
}

/*!
\overload

Sets the uniform variable array called \a name in the current
context to the \a count 3D vector elements of \a values.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValueArray(const char *name, const glm::vec3 *values, int count)
{
	setUniformValueArray(uniformLocation(name), values, count);
}

/*!
Sets the uniform variable array at \a location in the current
context to the \a count 4D vector elements of \a values.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValueArray(int location, const glm::vec4 *values, int count)
{
	if (location != -1)
		glUniform4fv(location, count, reinterpret_cast<const GLfloat *>(values));
}

/*!
\overload

Sets the uniform variable array called \a name in the current
context to the \a count 4D vector elements of \a values.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValueArray(const char *name, const glm::vec4 *values, int count)
{
	setUniformValueArray(uniformLocation(name), values, count);
}

// We have to repack matrix arrays from qreal to GLfloat.
#define setUniformMatrixArray(func,location,values,count,type,cols,rows) \
    if (location == -1 || count <= 0) \
        return; \
    if (sizeof(type) == sizeof(GLfloat) * cols * rows) { \
        func(location, count, GL_FALSE, \
             reinterpret_cast<const GLfloat *>(glm::value_ptr(values[0]))); \
    } else { \
        std::vector<GLfloat> temp(cols * rows * count); \
        for (int index = 0; index < count; ++index) { \
            for (int index2 = 0; index2 < (cols * rows); ++index2) { \
                temp.data()[cols * rows * index + index2] = \
                    glm::value_ptr(values[index])[index2]; \
            } \
        } \
        func(location, count, GL_FALSE, temp.data()); \
    }
#define setUniformGenericMatrixArray(colfunc,location,values,count,type,cols,rows) \
    if (location == -1 || count <= 0) \
        return; \
    if (sizeof(type) == sizeof(GLfloat) * cols * rows) { \
        const GLfloat *data = reinterpret_cast<const GLfloat *> \
            (values[0].constData());  \
        colfunc(location, count * cols, data); \
    } else { \
        QVarLengthArray<GLfloat> temp(cols * rows * count); \
        for (int index = 0; index < count; ++index) { \
            for (int index2 = 0; index2 < (cols * rows); ++index2) { \
                temp.data()[cols * rows * index + index2] = \
                    values[index].constData()[index2]; \
            } \
        } \
        colfunc(location, count * cols, temp.constData()); \
    }

/*!
Sets the uniform variable array at \a location in the current
context to the \a count 2x2 matrix elements of \a values.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValueArray(int location, const glm::mat2 *values, int count)
{
	setUniformMatrixArray
	(glUniformMatrix2fv, location, values, count, glm::mat2, 2, 2);
}

/*!
\overload

Sets the uniform variable array called \a name in the current
context to the \a count 2x2 matrix elements of \a values.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValueArray(const char *name, const glm::mat2 *values, int count)
{
	setUniformValueArray(uniformLocation(name), values, count);
}


/*!
Sets the uniform variable array at \a location in the current
context to the \a count 3x3 matrix elements of \a values.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValueArray(int location, const glm::mat3 *values, int count)
{
	setUniformMatrixArray
	(glUniformMatrix3fv, location, values, count, glm::mat3, 3, 3);
}

/*!
\overload

Sets the uniform variable array called \a name in the current
context to the \a count 3x3 matrix elements of \a values.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValueArray(const char *name, const glm::mat3 *values, int count)
{
	setUniformValueArray(uniformLocation(name), values, count);
}

/*!
Sets the uniform variable array at \a location in the current
context to the \a count 4x4 matrix elements of \a values.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValueArray(int location, const glm::mat4 *values, int count)
{
	setUniformMatrixArray
	(glUniformMatrix4fv, location, values, count, glm::mat4, 4, 4);
}

/*!
\overload

Sets the uniform variable array called \a name in the current
context to the \a count 4x4 matrix elements of \a values.

\sa setAttributeValue()
*/
void OpenGLShaderProgram::setUniformValueArray(const char *name, const glm::mat4 *values, int count)
{
	setUniformValueArray(uniformLocation(name), values, count);
}

/*!
Returns the hardware limit for how many vertices a geometry shader
can output.
*/
int OpenGLShaderProgram::maxGeometryOutputVertices() const
{
	GLint n = 0;
	glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES, &n);
	return n;
}

/*!
Use this function to specify to OpenGL the number of vertices in
a patch to \a count. A patch is a custom OpenGL primitive whose interpretation
is entirely defined by the tessellation shader stages. Therefore, calling
this function only makes sense when using a OpenGLShaderProgram
containing tessellation stage shaders. When using OpenGL tessellation,
the only primitive that can be rendered with \c{glDraw*()} functions is
\c{GL_PATCHES}.

This is equivalent to calling glPatchParameteri(GL_PATCH_VERTICES, count).

\note This modifies global OpenGL state and is not specific to this
OpenGLShaderProgram instance. You should call this in your render
function when needed, as OpenGLShaderProgram will not apply this for
you. This is purely a convenience function.

\sa patchVertexCount()
*/
void OpenGLShaderProgram::setPatchVertexCount(int count)
{
	glPatchParameteri(GL_PATCH_VERTICES, count);
}

/*!
Returns the number of vertices per-patch to be used when rendering.

\note This returns the global OpenGL state value. It is not specific to
this OpenGLShaderProgram instance.

\sa setPatchVertexCount()
*/
int OpenGLShaderProgram::patchVertexCount() const
{
	int patchVertices = 0;
	glGetIntegerv(GL_PATCH_VERTICES, &patchVertices);
	return patchVertices;
}

/*!
Sets the default outer tessellation levels to be used by the tessellation
primitive generator in the event that the tessellation control shader
does not output them to \a levels. For more details on OpenGL and Tessellation
shaders see \l{OpenGL Tessellation Shaders}.

The \a levels argument should be a QVector consisting of 4 floats. Not all
of the values make sense for all tessellation modes. If you specify a vector with
fewer than 4 elements, the remaining elements will be given a default value of 1.

\note This modifies global OpenGL state and is not specific to this
OpenGLShaderProgram instance. You should call this in your render
function when needed, as OpenGLShaderProgram will not apply this for
you. This is purely a convenience function.

\sa defaultOuterTessellationLevels(), setDefaultInnerTessellationLevels()
*/
void OpenGLShaderProgram::setDefaultOuterTessellationLevels(const std::vector<float> &levels)
{
	std::vector<float> tessLevels = levels;

	// Ensure we have the required 4 outer tessellation levels
	// Use default of 1 for missing entries (same as spec)
	const int argCount = 4;
	if (tessLevels.size() < argCount) {
		tessLevels.reserve(argCount);
		for (size_t i = tessLevels.size(); i < argCount; ++i)
			tessLevels.push_back(1.0f);
	}
	glPatchParameterfv(GL_PATCH_DEFAULT_OUTER_LEVEL, tessLevels.data());
}

/*!
Returns the default outer tessellation levels to be used by the tessellation
primitive generator in the event that the tessellation control shader
does not output them. For more details on OpenGL and Tessellation shaders see
\l{OpenGL Tessellation Shaders}.

Returns a QVector of floats describing the outer tessellation levels. The vector
will always have four elements but not all of them make sense for every mode
of tessellation.

\note This returns the global OpenGL state value. It is not specific to
this OpenGLShaderProgram instance.

\sa setDefaultOuterTessellationLevels(), defaultInnerTessellationLevels()
*/
std::vector<float> OpenGLShaderProgram::defaultOuterTessellationLevels() const
{
	std::vector<float> tessLevels(4, 1.0f);
	glGetFloatv(GL_PATCH_DEFAULT_OUTER_LEVEL, tessLevels.data());
	return tessLevels;
}

/*!
Sets the default outer tessellation levels to be used by the tessellation
primitive generator in the event that the tessellation control shader
does not output them to \a levels. For more details on OpenGL and Tessellation shaders see
\l{OpenGL Tessellation Shaders}.

The \a levels argument should be a QVector consisting of 2 floats. Not all
of the values make sense for all tessellation modes. If you specify a vector with
fewer than 2 elements, the remaining elements will be given a default value of 1.

\note This modifies global OpenGL state and is not specific to this
OpenGLShaderProgram instance. You should call this in your render
function when needed, as OpenGLShaderProgram will not apply this for
you. This is purely a convenience function.

\sa defaultInnerTessellationLevels(), setDefaultOuterTessellationLevels()
*/
void OpenGLShaderProgram::setDefaultInnerTessellationLevels(const std::vector<float> &levels)
{
	std::vector<float> tessLevels = levels;

	// Ensure we have the required 2 inner tessellation levels
	// Use default of 1 for missing entries (same as spec)
	const int argCount = 2;
	if (tessLevels.size() < argCount) {
		tessLevels.reserve(argCount);
		for (size_t i = tessLevels.size(); i < argCount; ++i)
			tessLevels.push_back(1.0f);
	}
	glPatchParameterfv(GL_PATCH_DEFAULT_INNER_LEVEL, tessLevels.data());
}

/*!
Returns the default inner tessellation levels to be used by the tessellation
primitive generator in the event that the tessellation control shader
does not output them. For more details on OpenGL and Tessellation shaders see
\l{OpenGL Tessellation Shaders}.

Returns a QVector of floats describing the inner tessellation levels. The vector
will always have two elements but not all of them make sense for every mode
of tessellation.

\note This returns the global OpenGL state value. It is not specific to
this OpenGLShaderProgram instance.

\sa setDefaultInnerTessellationLevels(), defaultOuterTessellationLevels()
*/
std::vector<float> OpenGLShaderProgram::defaultInnerTessellationLevels() const
{
	std::vector<float> tessLevels(2, 1.0f);
	glGetFloatv(GL_PATCH_DEFAULT_INNER_LEVEL, tessLevels.data());
	return tessLevels;
}

/*!
Returns \c true if shader programs of type \a type are supported on
this system; false otherwise.

The \a context is used to resolve the GLSL extensions.
If \a context is null, then QOpenGLContext::currentContext() is used.
*/
bool OpenGLShader::hasOpenGLShaders(ShaderType type)
{
	if ((type & ~(Geometry | Vertex | Fragment | TessellationControl | TessellationEvaluation | Compute)) || type == 0)
		return false;

	if (type & OpenGLShader::Geometry)
		return supportsGeometry();
	else if (type & (OpenGLShader::TessellationControl | OpenGLShader::TessellationEvaluation))
		return supportsTessellation();
	else if (type & OpenGLShader::Compute)
		return supportsCompute();

	// Unconditional support of vertex and fragment shaders implicitly assumes
	// a minimum OpenGL version of 2.0
	return true;
}




