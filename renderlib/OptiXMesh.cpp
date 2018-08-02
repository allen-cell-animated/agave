/* 
 * Copyright (c) 2017 NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if defined(_WIN32)
#define NOMINMAX
#endif
#include <optixu/optixu_math_namespace.h>

#include "OptiXMesh.h"
#include "Logging.h"

#include "cudarndr/BoundingBox.h"

#include "assimp/Importer.hpp"
#include "assimp/postprocess.h"
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

optix::Transform createSingleGeometryGroup(const aiScene* scene, optix::Context context, optix::Program meshIntersectProgram, optix::Program meshBboxProgram, optix::float3 *vertexMap,
	optix::float3 *normalMap, optix::uint3 *faceMap, unsigned int *materialsMap, optix::Material matl, glm::mat4& mtx) {

	std::vector<optix::GeometryInstance> gis;
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
		optix::Geometry geometry = context->createGeometry();
		geometry->setPrimitiveCount(mesh->mNumFaces);
		geometry->setIntersectionProgram(meshIntersectProgram);
		geometry->setBoundingBoxProgram(meshBboxProgram);
		geometry->setPrimitiveIndexOffset(faceOffset);

		optix::GeometryInstance gi = context->createGeometryInstance(geometry, &matl, &matl + 1);
		gis.push_back(gi);

		vertexOffset += mesh->mNumVertices;
		faceOffset += mesh->mNumFaces;

	}

	// add all geometry instances to a geometry group
	optix::Transform transform = context->createTransform();

	optix::GeometryGroup gg = context->createGeometryGroup();
	gg->setChildCount(static_cast<unsigned int>(gis.size()));
	for (unsigned i = 0u; i < gis.size(); i++) {
		gg->setChild(i, gis[i]);
	}
	optix::Acceleration a = context->createAcceleration("Trbvh");
	gg->setAcceleration(a);

	transform->setMatrix(false, glm::value_ptr(mtx), NULL);
	transform->setChild(gg);

	return transform;
}

optix::Transform loadAsset(const aiScene* scene, optix::Context context, glm::mat4& mtx)
{
	optix::Transform transformedggroup;

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
			optix::Buffer vertices = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, numVerts);
			optix::Buffer normals = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, numVerts);
			optix::Buffer faces = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, numFaces);
			// each face can have a different material...
			optix::Buffer materials = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, numFaces);

			// unused buffer
			optix::Buffer tbuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, 0);

			// create material
			optix::Program phong_ch = context->createProgramFromPTXFile("./ptx/objects-Debug/CudaPTX/phong.ptx", "closest_hit_radiance");
			optix::Program phong_ah = context->createProgramFromPTXFile("./ptx/objects-Debug/CudaPTX/phong.ptx", "any_hit_shadow");

			optix::Material matl = context->createMaterial();
			matl->setClosestHitProgram(0, phong_ch);
			matl->setAnyHitProgram(1, phong_ah);
			matl["Kd"]->setFloat(0.7f, 0.7f, 0.7f);
			matl["Ka"]->setFloat(1.0f, 1.0f, 1.0f);
			matl["Kr"]->setFloat(0.0f, 0.0f, 0.0f);
			matl["phong_exp"]->setFloat(1.0f);

			std::string triangle_mesh_ptx_path("./ptx/objects-Debug/CudaPTX/triangle_mesh.ptx");
			optix::Program meshIntersectProgram = context->createProgramFromPTXFile(triangle_mesh_ptx_path, "mesh_intersect");
			optix::Program meshBboxProgram = context->createProgramFromPTXFile(triangle_mesh_ptx_path, "mesh_bounds");

			optix::float3 *vertexMap = reinterpret_cast<optix::float3*>(vertices->map());
			optix::float3 *normalMap = reinterpret_cast<optix::float3*>(normals->map());
			optix::uint3 *faceMap = reinterpret_cast<optix::uint3*>(faces->map());
			unsigned int *materialsMap = static_cast<unsigned int*>(materials->map());

			context["vertex_buffer"]->setBuffer(vertices);
			context["normal_buffer"]->setBuffer(normals);
			context["index_buffer"]->setBuffer(faces);
			context["texcoord_buffer"]->setBuffer(tbuffer);
			context["material_buffer"]->setBuffer(materials);

			transformedggroup = createSingleGeometryGroup(scene, context, meshIntersectProgram, meshBboxProgram, vertexMap,
				normalMap, faceMap, materialsMap, matl, mtx);

			vertices->unmap();
			normals->unmap();
			faces->unmap();
			materials->unmap();

		}

	}
	return transformedggroup;
}
