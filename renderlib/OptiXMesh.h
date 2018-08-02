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


#pragma once

#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

#include "assimp/Importer.hpp"

#include "glm.h"

#include <memory>

struct aiScene;

//------------------------------------------------------------------------------
//
// OptiX mesh consisting of a single geometry instance with one or more
// materials.
//
// Mesh buffer variables are set on Geometry node:
//   vertex_buffer  : float3 vertex positions
//   normal_buffer  : float3 per vertex normals, may be zero length 
//   texcoord_buffer: float2 vertex texture coordinates, may be zero length
//   index_buffer   : int3 indices shared by vertex, normal, texcoord buffers 
//   material_buffer: int indices into material list
//
//------------------------------------------------------------------------------
struct OptiXMesh
{
	OptiXMesh(std::shared_ptr<Assimp::Importer> cpumesh, optix::Context context, glm::mat4& mtx);
	// Input
	std::shared_ptr<Assimp::Importer> _cpumesh;

	optix::Context _context;       // required

	optix::Program _intersection;  // optional 
	optix::Program _bounds;        // optional

	optix::Program _closest_hit;   // optional multi matl override
	optix::Program _any_hit;       // optional

	// Output
	optix::Buffer _vertices;
	optix::Buffer _normals;
	optix::Buffer _faces;
	// each face can have a different material...
	optix::Buffer _materials;
	// unused buffer (uv)
	optix::Buffer _tbuffer;

	// will be set by app, and ignore what comes out of the assimp importer.
	optix::Material _material;

	// top node for this mesh
	optix::Transform _transform;
	// child of the transform
	optix::GeometryGroup _geometrygroup;
	// children of the geometry group
	std::vector<optix::GeometryInstance> _gis;


	//optix::float3                bbox_min;
	//optix::float3                bbox_max;

private:
	bool loadAsset(glm::mat4& mtx);
	void OptiXMesh::createSingleGeometryGroup(const aiScene* scene, optix::Program meshIntersectProgram, optix::Program meshBboxProgram, optix::float3 *vertexMap,
		optix::float3 *normalMap, optix::uint3 *faceMap, unsigned int *materialsMap, optix::Material matl, glm::mat4& mtx);
};
