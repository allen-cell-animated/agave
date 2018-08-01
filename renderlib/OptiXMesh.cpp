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

optix::Group createSingleGeometryGroup(const aiScene* scene, optix::Context context, optix::Program meshIntersectProgram, optix::Program meshBboxProgram, optix::float3 *vertexMap,
	optix::float3 *normalMap, optix::uint3 *faceMap, unsigned int *materialsMap, optix::Material matl, CBoundingBox& bb) {

	optix::Group group = context->createGroup();
	optix::Acceleration accel = context->createAcceleration("Trbvh");
	group->setAcceleration(accel);
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

			bb += glm::vec3(pos.x, pos.y, pos.z);
			vertexMap[i + vertexOffset] = optix::make_float3(pos.x, pos.y, pos.z);// +aabb.center();
			normalMap[i + vertexOffset] = optix::normalize(optix::make_float3(norm.x, norm.y, norm.z));
			materialsMap[i + vertexOffset] = 0u;

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

	//printf("VertexOffset: %d\nFaceOffset: %d\n", vertexOffset, faceOffset);
	//printf("BBOX: X:(%f,%f)  Y:(%f,%f)  Z:(%f,%f)\n", bb.GetMinP().x, bb.GetMaxP().x, bb.GetMinP().y, bb.GetMaxP().y, bb.GetMinP().z, bb.GetMaxP().z);
	// add all geometry instances to a geometry group
	optix::GeometryGroup gg = context->createGeometryGroup();
	gg->setChildCount(static_cast<unsigned int>(gis.size()));
	for (unsigned i = 0u; i < gis.size(); i++) {
		gg->setChild(i, gis[i]);
	}
	optix::Acceleration a = context->createAcceleration("Trbvh");
	gg->setAcceleration(a);

	group->setChildCount(1);
	group->setChild(0, gg);

	return group;
}

int loadAsset(const aiScene* scene, optix::Context context, RTgroup* o_group, CBoundingBox& bb)
{
	if (scene) {
		//getBoundingBox(&scene_min, &scene_max);
		//scene_center.x = (scene_min.x + scene_max.x) / 2.0f;
		//scene_center.y = (scene_min.y + scene_max.y) / 2.0f;
		//scene_center.z = (scene_min.z + scene_max.z) / 2.0f;

		//float3 optixMin = { scene_min.x, scene_min.y, scene_min.z };
		//float3 optixMax = { scene_max.x, scene_max.y, scene_max.z };
		//aabb.set(optixMin, optixMax);

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
			optix::Buffer materials = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, numVerts);

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

			optix::Group group = createSingleGeometryGroup(scene, context, meshIntersectProgram, meshBboxProgram, vertexMap,
				normalMap, faceMap, materialsMap, matl, bb);

			context["top_object"]->set(group);
			context["top_shadower"]->set(group);
			context["max_depth"]->setInt(100);
			context["radiance_ray_type"]->setUint(0);
			context["shadow_ray_type"]->setUint(1);
			//context["scene_epsilon"]->setFloat(1.e-4f);
			context["importance_cutoff"]->setFloat(0.01f);
			context["ambient_light_color"]->setFloat(0.31f, 0.33f, 0.28f);

			vertices->unmap();
			normals->unmap();
			faces->unmap();
			materials->unmap();

			*o_group = group->get();
		}

		return 0;
	}
	return 1;
}
