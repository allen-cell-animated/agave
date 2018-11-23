#if defined(_WIN32)
#define NOMINMAX
#endif

#include "RenderGLOptix.h"

#include "glad/glad.h"
#include "glm.h"

#include "gl/Util.h"
#include "CCamera.h"
#include "ImageXYZC.h"
#include "Logging.h"

#include "OptiXMesh.h"

#include <optix_gl_interop.h>

#include <array>

// Error check/report helper for users of the C API
#define RT_CHECK_ERROR( func )                                     \
  do {                                                             \
    RTresult code = func;                                          \
    if( code != RT_SUCCESS ) {                                     \
	    const char* message;                                       \
		rtContextGetErrorString(_context, code, &message);         \
		LOG_ERROR << message;                                      \
    }                                                              \
  } while(0)

struct BasicLight
{
	float pos[3];
	float color[3];
	int    casts_shadow;
	int    padding;      // make this structure 32 bytes -- powers of two are your friend!
};

RenderGLOptix::RenderGLOptix(RenderSettings* rs)
	: _renderSettings(rs),
	_w(0),
	_h(0),
	_scene(nullptr),
	_gpuBytes(0),
	_context(0),
	_light_buffer(0)
{
}


RenderGLOptix::~RenderGLOptix()
{
}

GLuint _pixelBuffer;
GLuint _hdrTexture;

void RenderGLOptix::initialize(uint32_t w, uint32_t h)
{
	_imagequad = new RectImage2D();

	glGenBuffers(1, &_pixelBuffer);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pixelBuffer);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, w*h * sizeof(float) * 4, 0, GL_STREAM_READ);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	check_gl("create pbo");

	glGenTextures(1, &_hdrTexture);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, _hdrTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glBindTexture(GL_TEXTURE_2D, 0);
	check_gl("create buffer texture");

	//RT_CHECK_ERROR( rtContextCreate( &_context ) );
	_ctx = optix::Context::create();
	_context = _ctx->get();

	/* Create our objects and set state */
	
	optix::Group group = _ctx->createGroup();
	_topGroup = group->get();
	optix::Acceleration accel = _ctx->createAcceleration("Trbvh");
	group->setAcceleration(accel);
	_ctx["top_object"]->set(group);
	_ctx["top_shadower"]->set(group);
	_ctx["max_depth"]->setInt(100);
	//_ctx["scene_epsilon"]->setFloat(1.e-4f);
	_ctx["importance_cutoff"]->setFloat(0.01f);
	_ctx["ambient_light_color"]->setFloat(0.31f, 0.33f, 0.28f);

	RT_CHECK_ERROR( rtContextSetRayTypeCount( _context, 2 ) );
	_ctx["radiance_ray_type"]->setUint(0);
	_ctx["shadow_ray_type"]->setUint(1);

	RT_CHECK_ERROR( rtContextSetEntryPointCount( _context, 1 ) );

	RT_CHECK_ERROR(rtBufferCreateFromGLBO(_context, RT_BUFFER_OUTPUT, _pixelBuffer, &_buffer));
	//RT_CHECK_ERROR( rtBufferCreate( _context, RT_BUFFER_OUTPUT, &_buffer ) );
	RT_CHECK_ERROR( rtBufferSetFormat( _buffer, RT_FORMAT_FLOAT4 ) );
	RT_CHECK_ERROR( rtBufferSetSize2D( _buffer, w, h ) );
	//RT_CHECK_ERROR( rtBufferGLRegister(_buffer) );

	RT_CHECK_ERROR( rtContextDeclareVariable( _context, "result_buffer", &_result_buffer ) );
	RT_CHECK_ERROR( rtVariableSetObject( _result_buffer, _buffer ) );

	char* path_to_ptx = "./ptx/objects-Debug/CudaPTX/hello.ptx";
	//char* path_to_ptx = "./ptx/objects-Release/CudaPTX/hello.ptx";
	RT_CHECK_ERROR(rtProgramCreateFromPTXFile(_context, path_to_ptx, "pinhole_camera", &_ray_gen_program));
	RT_CHECK_ERROR(rtProgramCreateFromPTXFile(_context, path_to_ptx, "miss", &_miss_program));
	RT_CHECK_ERROR(rtProgramCreateFromPTXFile(_context, path_to_ptx, "exception", &_exception_program));

	// create material
	_phong_closesthit_program = _ctx->createProgramFromPTXFile("./ptx/objects-Debug/CudaPTX/phong.ptx", "closest_hit_radiance");
	_phong_anyhit_program = _ctx->createProgramFromPTXFile("./ptx/objects-Debug/CudaPTX/phong.ptx", "any_hit_shadow");
	std::string triangle_mesh_ptx_path("./ptx/objects-Debug/CudaPTX/triangle_mesh.ptx");
	_mesh_intersect_program = _ctx->createProgramFromPTXFile(triangle_mesh_ptx_path, "mesh_intersect");
	_mesh_boundingbox_program = _ctx->createProgramFromPTXFile(triangle_mesh_ptx_path, "mesh_bounds");

	RT_CHECK_ERROR(rtProgramDeclareVariable(_ray_gen_program, "draw_color", &_draw_color));
	RT_CHECK_ERROR(rtProgramDeclareVariable(_ray_gen_program, "scene_epsilon", &_scene_epsilon));
	RT_CHECK_ERROR(rtProgramDeclareVariable(_ray_gen_program, "eye", &_eye));
	RT_CHECK_ERROR(rtProgramDeclareVariable(_ray_gen_program, "U", &_U));
	RT_CHECK_ERROR(rtProgramDeclareVariable(_ray_gen_program, "V", &_V));
	RT_CHECK_ERROR(rtProgramDeclareVariable(_ray_gen_program, "W", &_W));

	RT_CHECK_ERROR(rtVariableSet3f(_draw_color, 0.462f, 0.725f, 0.0f));
	RT_CHECK_ERROR(rtVariableSet1f(_scene_epsilon, 1.e-4f));

	RT_CHECK_ERROR(rtContextSetRayGenerationProgram(_context, 0, _ray_gen_program));
	RT_CHECK_ERROR(rtContextSetMissProgram(_context, 0, _miss_program));
	RT_CHECK_ERROR(rtContextSetExceptionProgram(_context, 0, _exception_program));

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	check_gl("init gl state");

	// Size viewport
	resize(w,h);
}

glm::vec3 nextColor() {
	static float currentHue = 0.0f;
	glm::vec3 v = glm::rgbColor(glm::vec3(currentHue * 360.0, 1.0f, 0.5f));
	currentHue += 0.618033988749895f;
	currentHue = std::fmod(currentHue, 1.0f);
	return v;
}

void RenderGLOptix::initOptixMesh() {
	glm::mat4 mtx(1.0);

	TriMeshPhongPrograms prg;
	prg._closestHit = _phong_closesthit_program;
	prg._anyHit = _phong_anyhit_program;
	prg._boundingBox = _mesh_boundingbox_program;
	prg._intersect = _mesh_intersect_program;

	// remove all children and rebuild.
	//RT_CHECK_ERROR(rtGroupSetChildCount(_topGroup, 0));
	//_optixmeshes.clear();

	for (int i = 0; i < _scene->m_meshes.size(); ++i) {
		// see if this mesh already exists in renderable form?
		bool found = false;
		for (int j = 0; j < _optixmeshes.size(); ++j) {
			if (_optixmeshes[j]->_cpumesh == _scene->m_meshes[i]) {
				found = true;
				LOG_DEBUG << "found mesh already";
				break;
			}
		}
		if (found) {
			continue;
		}

		optixMeshMaterial materialdesc;
		materialdesc._reflectivity = nextColor();
		OptiXMesh* optixmesh = new OptiXMesh(_scene->m_meshes[i], _ctx, prg, mtx, &materialdesc);

		_optixmeshes.push_back(std::shared_ptr<OptiXMesh>(optixmesh));

		optix::Transform transformedggroup = optixmesh->_transform;
		//optix::Transform transformedggroup = loadAsset(_scene->_meshes[0]->GetScene(), _ctx, mtx);
		if (transformedggroup) {
			unsigned int index = 0;
			RT_CHECK_ERROR(rtGroupGetChildCount(_topGroup, &index));
			RT_CHECK_ERROR(rtGroupSetChildCount(_topGroup, index + 1));
			RT_CHECK_ERROR(rtGroupSetChild(_topGroup, index, transformedggroup->get()));

		}
	}

	{

		BasicLight lights[] = {
			{ { 79.0f, 6.0f, -16.0f },{ 1.0f, 1.0f, 1.0f }, 1 }
		};

		RT_CHECK_ERROR(rtBufferCreate(_context, RT_BUFFER_INPUT, &_light_buffer));
		RT_CHECK_ERROR(rtBufferSetFormat(_light_buffer, RT_FORMAT_USER));
		RT_CHECK_ERROR(rtBufferSetElementSize(_light_buffer, sizeof(BasicLight)));
		RT_CHECK_ERROR(rtBufferSetSize1D(_light_buffer, sizeof(lights) / sizeof(lights[0])));

		void* mapped = nullptr;
		RT_CHECK_ERROR(rtBufferMap(_light_buffer, &mapped));
		memcpy(mapped, lights, sizeof(lights));
		RT_CHECK_ERROR(rtBufferUnmap(_light_buffer));

		// descend into group and get material, then get closest hit program to set light_buffer ?
		//unsigned int count;
		//rtGroupGetChildCount(topgroup, &count);
		//RTobject childob;
		//rtGroupGetChild(topgroup, 0, &childob);

		RT_CHECK_ERROR(rtContextDeclareVariable(_context, "lights", &_lightsvar));
		RT_CHECK_ERROR(rtVariableSetObject(_lightsvar, _light_buffer));
	}

}

void RenderGLOptix::doRender(const CCamera& camera) {
	if (!_scene || _scene->m_meshes.empty()) {
		return;
	}
	if (_renderSettings->m_DirtyFlags.HasFlag(MeshDirty) && !_light_buffer) {
		initOptixMesh();
		// we have set up everything there is to do before rendering
		//_status.SetRenderBegin();
	}

	static bool cameraInit = false;
	if (!cameraInit && _scene) {

		const_cast<CCamera*>(&camera)->m_SceneBoundingBox.m_MinP = _scene->m_boundingBox.GetMinP();
		const_cast<CCamera*>(&camera)->m_SceneBoundingBox.m_MaxP = _scene->m_boundingBox.GetMaxP();
		// reposition to face image
		const_cast<CCamera*>(&camera)->SetViewMode(ViewModeFront);
		cameraInit = true;
	}
	//camera.m_SceneBoundingBox = _scene->_boundingBox;
	//if (_renderSettings->m_DirtyFlags.HasFlag(CameraDirty))
	{
		RT_CHECK_ERROR(rtVariableSet3f(_eye, camera.m_From.x, camera.m_From.y, camera.m_From.z));
		RT_CHECK_ERROR(rtVariableSet3f(_U, camera.m_U.x, camera.m_U.y, camera.m_U.z));
		RT_CHECK_ERROR(rtVariableSet3f(_V, camera.m_V.x, camera.m_V.y, camera.m_V.z));
		RT_CHECK_ERROR(rtVariableSet3f(_W, camera.m_N.x, camera.m_N.y, camera.m_N.z));
	}
	if (_renderSettings->m_DirtyFlags.HasFlag(LightsDirty))
	{
		for (int i = 0; i < _scene->m_lighting.m_NoLights; ++i) {
			_scene->m_lighting.m_Lights[i].Update(_scene->m_boundingBox);
		}
		//printf("LIGHT (%f, %f, %f)\n", _scene->_lighting.m_Lights[1].m_P.x, _scene->_lighting.m_Lights[1].m_P.y, _scene->_lighting.m_Lights[1].m_P.z);

	}
	BasicLight lights[] = {
		{ { 79.0f, 6.0f, -16.0f },{ 1.0f, 1.0f, 1.0f }, 1 }
	};
	lights[0].pos[0] = _scene->m_lighting.m_Lights[1].m_P.x;
	lights[0].pos[1] = _scene->m_lighting.m_Lights[1].m_P.y;
	lights[0].pos[2] = _scene->m_lighting.m_Lights[1].m_P.z;
	lights[0].color[0] = _scene->m_lighting.m_Lights[1].m_Color.x;
	lights[0].color[1] = _scene->m_lighting.m_Lights[1].m_Color.y;
	lights[0].color[2] = _scene->m_lighting.m_Lights[1].m_Color.z;

	void* mapped = nullptr;
	// if number of lights changed, might need to update size of this buffer...
	//RT_CHECK_ERROR(rtBufferSetSize1D(_light_buffer, 1));
	RT_CHECK_ERROR(rtBufferMap(_light_buffer, &mapped));
	memcpy(mapped, lights, sizeof(lights));
	RT_CHECK_ERROR(rtBufferUnmap(_light_buffer));
	//RT_CHECK_ERROR(rtVariableSetObject(_lightsvar, _light_buffer));


	// At this point, all dirty flags should have been taken care of, since the flags in the original scene are now cleared
	_renderSettings->m_DirtyFlags.ClearAllFlags();

	/* Run */
	RT_CHECK_ERROR( rtContextValidate( _context ) );
	RT_CHECK_ERROR( rtContextLaunch2D( _context, 0 /* entry point */, _w, _h ) );

	// get the render result (in _pixelBuffer) into a texture (_hdrTexture)
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, _hdrTexture);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pixelBuffer);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, (GLsizei)_w, (GLsizei)_h, 0, GL_RGBA, GL_FLOAT, (void*)0); // RGBA32F from byte offset 0 in the pixel unpack buffer.
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void RenderGLOptix::render(const CCamera& camera)
{
	// draw to _fbtex
	doRender(camera);

	// put _fbtex to main render target
	drawImage();
}

void RenderGLOptix::drawImage() {
#if 0
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pixelBuffer);
	float* ptr = (float*)glMapBuffer(GL_PIXEL_UNPACK_BUFFER, GL_READ_ONLY);
	// look at pixel.
	LOG_DEBUG << ptr[0] << ", " << ptr[1] << ", " << ptr[2] << ", " << ptr[3];
	float* rgba = new float[_w*_h * 4];
	memcpy(rgba, ptr, _w*_h * 4 * sizeof(float));
	glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
#endif

	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glDisable(GL_BLEND);
	_imagequad->draw(_hdrTexture);

#if 0
    // Query buffer information
	// this had better be equal to _w, _h !!
    RTsize buffer_width_rts, buffer_height_rts;
    rtBufferGetSize2D(_buffer, &buffer_width_rts, &buffer_height_rts );
    uint32_t width  = static_cast<int>(buffer_width_rts);
    uint32_t height = static_cast<int>(buffer_height_rts);
	assert(width == _w);
	assert(height == _h);
	RTformat buffer_format;
	rtBufferGetFormat(_buffer, &buffer_format);
    
    GLboolean use_SRGB = GL_FALSE;
    if( buffer_format == RT_FORMAT_FLOAT4 || buffer_format == RT_FORMAT_FLOAT3 )
    {
//        glGetBooleanv( GL_FRAMEBUFFER_SRGB_CAPABLE_EXT, &use_SRGB );
//        if( use_SRGB )
//            glEnable(GL_FRAMEBUFFER_SRGB_EXT);
    }

    // Check if we have a GL interop display buffer
	unsigned int pboId = 0;
	rtBufferGetGLBOId(_buffer, &pboId);
    if( pboId )
    {
        static unsigned int gl_tex_id = 0;
        if( !gl_tex_id )
        {
            glGenTextures( 1, &gl_tex_id );
            glBindTexture( GL_TEXTURE_2D, gl_tex_id );
			check_gl("create pbo texture");

            // Change these to GL_LINEAR for super- or sub-sampling
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

            // GL_CLAMP_TO_EDGE for linear filtering, not relevant for nearest.
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }

        glBindTexture( GL_TEXTURE_2D, gl_tex_id );

        // send PBO to texture
        glBindBuffer( GL_PIXEL_UNPACK_BUFFER, pboId );

		RTsize elmt_size;
		rtBufferGetElementSize(_buffer, &elmt_size);
        if      ( elmt_size % 8 == 0)
			glPixelStorei(GL_UNPACK_ALIGNMENT, 8);
        else if ( elmt_size % 4 == 0) 
			glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        else if ( elmt_size % 2 == 0) 
			glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
        else                          
			glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        if( buffer_format == RT_FORMAT_UNSIGNED_BYTE4)
            glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, 0);
        else if(buffer_format == RT_FORMAT_FLOAT4)
            glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, 0);
        else if(buffer_format == RT_FORMAT_FLOAT3)
            glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB32F, width, height, 0, GL_RGB, GL_FLOAT, 0);
        else if(buffer_format == RT_FORMAT_FLOAT)
            glTexImage2D( GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, 0);
        else
          throw ( "Unknown buffer format" );
		check_gl("update pbo texture");

        glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );
		_imagequad->draw(gl_tex_id);
    }
    else
    {
		GLvoid* imageData;
		rtBufferMapEx(_buffer, RT_BUFFER_MAP_READ, 0, NULL, &imageData);
        GLenum gl_data_type = GL_FALSE;
        GLenum gl_format = GL_FALSE;

        switch (buffer_format)
        {
            case RT_FORMAT_UNSIGNED_BYTE4:
                gl_data_type = GL_UNSIGNED_BYTE;
                gl_format    = GL_BGRA;
                break;

            case RT_FORMAT_FLOAT:
                gl_data_type = GL_FLOAT;
                gl_format    = GL_RED;
                break;

            case RT_FORMAT_FLOAT3:
                gl_data_type = GL_FLOAT;
                gl_format    = GL_RGB;
                break;

            case RT_FORMAT_FLOAT4:
                gl_data_type = GL_FLOAT;
                gl_format    = GL_RGBA;
                break;

            default:
                fprintf(stderr, "Unrecognized buffer data type or format.\n");
                exit(2);
                break;
        }

		RTsize elmt_size = 0;
		rtBufferGetElementSize(_buffer, &elmt_size);
        int align = 1;
        if      ((elmt_size % 8) == 0) align = 8; 
        else if ((elmt_size % 4) == 0) align = 4;
        else if ((elmt_size % 2) == 0) align = 2;
        glPixelStorei(GL_UNPACK_ALIGNMENT, align);

#if 0
        glDrawPixels(
                static_cast<GLsizei>( width ),
                static_cast<GLsizei>( height ),
                gl_format,
                gl_data_type,
                imageData
                );
#endif
		rtBufferUnmapEx(_buffer, 0);
    }
#endif

//    if ( use_SRGB )
  //      glDisable(GL_FRAMEBUFFER_SRGB_EXT);

}

void RenderGLOptix::cleanUpResources() {
	// remove all children.
	RT_CHECK_ERROR(rtGroupSetChildCount(_topGroup, 0));
	_optixmeshes.clear();


	RT_CHECK_ERROR( rtBufferDestroy( _buffer ) );
	RT_CHECK_ERROR( rtProgramDestroy( _ray_gen_program ) );
	RT_CHECK_ERROR( rtContextDestroy( _context ) );	

	glDeleteBuffers(1, &_pixelBuffer);
	_pixelBuffer = 0;

	delete _imagequad;
	_imagequad = nullptr;
}

void RenderGLOptix::resize(uint32_t w, uint32_t h)
{
	//w = 8; h = 8;
	glViewport(0, 0, w, h);
	if ((_w == w) && (_h == h)) {
		return;
	}


	RT_CHECK_ERROR(rtBufferSetSize2D(_buffer, w, h));
	RT_CHECK_ERROR(rtBufferGLUnregister(_buffer));
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pixelBuffer);
	glBufferData(GL_PIXEL_UNPACK_BUFFER, w*h * sizeof(float) * 4, 0, GL_STREAM_DRAW);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	check_gl("resize pbo");
	RT_CHECK_ERROR(rtBufferGLRegister(_buffer));

	LOG_DEBUG << "Resized window to " << w << " x " << h;
	_w = w;
	_h = h;
}

RenderParams& RenderGLOptix::renderParams() {
	return _renderParams;
}
Scene* RenderGLOptix::scene() {
	return _scene;
}
void RenderGLOptix::setScene(Scene* s) {
	_scene = s;
}

size_t RenderGLOptix::getGpuBytes() {
	return 0;
}
