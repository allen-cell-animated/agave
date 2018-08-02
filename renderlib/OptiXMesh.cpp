#if defined(_WIN32)
#define NOMINMAX
#endif
#include <optixu/optixu_math_namespace.h>

#include "OptiXMesh.h"
#include "Logging.h"

#include "cudarndr/BoundingBox.h"

#include "assimp/scene.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace optix {
  float3 make_float3( const float* a )
  {
    return make_float3( a[0], a[1], a[2] );
  }
}

OptiXMesh::OptiXMesh(std::shared_ptr<Assimp::Importer> cpumesh, optix::Context context, glm::mat4& mtx)
{
	_cpumesh = cpumesh;
	_context = context;
	bool ok = loadAsset(mtx);
}

bool OptiXMesh::loadAsset(glm::mat4& mtx)
{
	const aiScene* scene = _cpumesh->GetScene();
	if (scene) {
		unsigned int numVerts = 0;
		unsigned int numFaces = 0;

		if (scene->mNumMeshes > 0) {
			// printf("Number of meshes: %d\n", scene->mNumMeshes);

			// get the running total number of vertices & faces for all meshes
			for (unsigned int i = 0; i < scene->mNumMeshes; i++) {
				numVerts += scene->mMeshes[i]->mNumVertices;
				numFaces += scene->mMeshes[i]->mNumFaces;
			}
			//printf("Found %d Vertices and %d Faces\n", numVerts, numFaces);

			// set up buffers
			_vertices = _context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, numVerts);
			_normals = _context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, numVerts);
			_faces = _context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, numFaces);
			// each face can have a different material...
			_materials = _context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, numFaces);

			// unused buffer
			_tbuffer = _context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, 0);

			// create material
			optix::Program phong_ch = _context->createProgramFromPTXFile("./ptx/objects-Debug/CudaPTX/phong.ptx", "closest_hit_radiance");
			optix::Program phong_ah = _context->createProgramFromPTXFile("./ptx/objects-Debug/CudaPTX/phong.ptx", "any_hit_shadow");

			_material = _context->createMaterial();
			_material->setClosestHitProgram(0, phong_ch);
			_material->setAnyHitProgram(1, phong_ah);
			_material["Kd"]->setFloat(0.7f, 0.7f, 0.7f);
			_material["Ka"]->setFloat(1.0f, 1.0f, 1.0f);
			_material["Kr"]->setFloat(0.0f, 0.0f, 0.0f);
			_material["phong_exp"]->setFloat(1.0f);

			std::string triangle_mesh_ptx_path("./ptx/objects-Debug/CudaPTX/triangle_mesh.ptx");
			optix::Program meshIntersectProgram = _context->createProgramFromPTXFile(triangle_mesh_ptx_path, "mesh_intersect");
			optix::Program meshBboxProgram = _context->createProgramFromPTXFile(triangle_mesh_ptx_path, "mesh_bounds");

			optix::float3 *vertexMap = reinterpret_cast<optix::float3*>(_vertices->map());
			optix::float3 *normalMap = reinterpret_cast<optix::float3*>(_normals->map());
			optix::uint3 *faceMap = reinterpret_cast<optix::uint3*>(_faces->map());
			unsigned int *materialsMap = static_cast<unsigned int*>(_materials->map());

			_context["vertex_buffer"]->setBuffer(_vertices);
			_context["normal_buffer"]->setBuffer(_normals);
			_context["index_buffer"]->setBuffer(_faces);
			_context["texcoord_buffer"]->setBuffer(_tbuffer);
			_context["material_buffer"]->setBuffer(_materials);

			createSingleGeometryGroup(scene, meshIntersectProgram, meshBboxProgram, vertexMap,
				normalMap, faceMap, materialsMap, _material, mtx);

			_vertices->unmap();
			_normals->unmap();
			_faces->unmap();
			_materials->unmap();

			return true;
		}
		return false;
	}
	return false;
}

void OptiXMesh::createSingleGeometryGroup(const aiScene* scene, optix::Program meshIntersectProgram, optix::Program meshBboxProgram, optix::float3 *vertexMap,
	optix::float3 *normalMap, optix::uint3 *faceMap, unsigned int *materialsMap, optix::Material matl, glm::mat4& mtx) {

	unsigned int vertexOffset = 0u;
	unsigned int faceOffset = 0u;

	for (unsigned int m = 0; m < scene->mNumMeshes; m++) {
		aiMesh *mesh = scene->mMeshes[m];
		if (!mesh->HasPositions()) {
			throw std::runtime_error("Mesh contains zero vertex positions");
		}
		if (!mesh->HasNormals()) {
			throw std::runtime_error("Mesh contains zero vertex normals");
		}

		//printf("Mesh #%d\n\tNumVertices: %d\n\tNumFaces: %d\n", m, mesh->mNumVertices, mesh->mNumFaces);

		// add points           
		for (unsigned int i = 0u; i < mesh->mNumVertices; i++) {
			aiVector3D pos = mesh->mVertices[i];
			aiVector3D norm = mesh->mNormals[i];

			vertexMap[i + vertexOffset] = optix::make_float3(pos.x, pos.y, pos.z);// +aabb.center();
			normalMap[i + vertexOffset] = optix::normalize(optix::make_float3(norm.x, norm.y, norm.z));

		}

		// add faces
		for (unsigned int i = 0u; i < mesh->mNumFaces; i++) {

			aiFace face = mesh->mFaces[i];

			// add triangles
			if (face.mNumIndices == 3) {
				faceMap[i + faceOffset] = optix::make_uint3(face.mIndices[0], face.mIndices[1], face.mIndices[2]);
			}
			else {
				//printf("face indices != 3\n");
				faceMap[i + faceOffset] = optix::make_uint3(-1);
			}
			materialsMap[i + faceOffset] = 0u;
		}

		// create geometry
		optix::Geometry geometry = _context->createGeometry();
		geometry->setPrimitiveCount(mesh->mNumFaces);
		geometry->setIntersectionProgram(meshIntersectProgram);
		geometry->setBoundingBoxProgram(meshBboxProgram);
		geometry->setPrimitiveIndexOffset(faceOffset);

		optix::GeometryInstance gi = _context->createGeometryInstance(geometry, &matl, &matl + 1);
		_gis.push_back(gi);

		vertexOffset += mesh->mNumVertices;
		faceOffset += mesh->mNumFaces;

	}

	// add all geometry instances to a geometry group
	_transform = _context->createTransform();

	_geometrygroup = _context->createGeometryGroup();
	_geometrygroup->setChildCount(static_cast<unsigned int>(_gis.size()));
	for (unsigned i = 0u; i < _gis.size(); i++) {
		_geometrygroup->setChild(i, _gis[i]);
	}
	optix::Acceleration a = _context->createAcceleration("Trbvh");
	_geometrygroup->setAcceleration(a);

	_transform->setMatrix(false, glm::value_ptr(mtx), NULL);
	_transform->setChild(_geometrygroup);
}
