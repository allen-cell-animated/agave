#if defined(_WIN32)
#define NOMINMAX
#endif
#include <optixu/optixu_math_namespace.h>

#include "Logging.h"
#include "OptiXMesh.h"

#include "cudarndr/BoundingBox.h"

#include "assimp/scene.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace optix {
float3
make_float3(const float* a)
{
  return make_float3(a[0], a[1], a[2]);
}
}

OptiXMesh::OptiXMesh(std::shared_ptr<Assimp::Importer> cpumesh,
                     optix::Context context,
                     TriMeshPhongPrograms& programs,
                     glm::mat4& mtx,
                     optixMeshMaterial* materialdesc)
{
  m_cpumesh = cpumesh;
  m_context = context;
  bool ok = loadAsset(programs, mtx, materialdesc);
}

bool
OptiXMesh::loadAsset(TriMeshPhongPrograms& programs, glm::mat4& mtx, optixMeshMaterial* materialdesc)
{
  const aiScene* scene = m_cpumesh->GetScene();
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
      // printf("Found %d Vertices and %d Faces\n", numVerts, numFaces);

      // set up buffers
      m_vertices = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, numVerts);
      m_normals = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT3, numVerts);
      m_faces = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, numFaces);
      // each face can have a different material...
      m_materials = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT, numFaces);

      // unused buffer
      m_tbuffer = m_context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT2, 0);

      // create material
      m_material = m_context->createMaterial();
      m_material->setClosestHitProgram(0, programs.m_closestHit);
      m_material->setAnyHitProgram(1, programs.m_anyHit);
      m_material["Kd"]->setFloat(
        materialdesc->m_reflectivity.x, materialdesc->m_reflectivity.y, materialdesc->m_reflectivity.z);
      m_material["Ka"]->setFloat(
        materialdesc->m_reflectivity.x, materialdesc->m_reflectivity.y, materialdesc->m_reflectivity.z);
      if (materialdesc->m_dielectric) {
        m_material["Kr"]->setFloat(
          materialdesc->m_reflectivity.x, materialdesc->m_reflectivity.y, materialdesc->m_reflectivity.z);
      } else {
        m_material["Kr"]->setFloat(0.0f, 0.0f, 0.0f);
      }
      m_material["phong_exp"]->setFloat(1.0f - materialdesc->m_roughness);

      optix::float3* vertexMap = reinterpret_cast<optix::float3*>(m_vertices->map());
      optix::float3* normalMap = reinterpret_cast<optix::float3*>(m_normals->map());
      optix::uint3* faceMap = reinterpret_cast<optix::uint3*>(m_faces->map());
      unsigned int* materialsMap = static_cast<unsigned int*>(m_materials->map());

      createSingleGeometryGroup(scene, programs, vertexMap, normalMap, faceMap, materialsMap, m_material, mtx);

      m_vertices->unmap();
      m_normals->unmap();
      m_faces->unmap();
      m_materials->unmap();

      return true;
    }
    return false;
  }
  return false;
}

void
OptiXMesh::createSingleGeometryGroup(const aiScene* scene,
                                     TriMeshPhongPrograms& programs,
                                     optix::float3* vertexMap,
                                     optix::float3* normalMap,
                                     optix::uint3* faceMap,
                                     unsigned int* materialsMap,
                                     optix::Material matl,
                                     glm::mat4& mtx)
{

  unsigned int vertexOffset = 0u;
  unsigned int faceOffset = 0u;

  for (unsigned int m = 0; m < scene->mNumMeshes; m++) {
    aiMesh* mesh = scene->mMeshes[m];
    if (!mesh->HasPositions()) {
      throw std::runtime_error("Mesh contains zero vertex positions");
    }
    if (!mesh->HasNormals()) {
      throw std::runtime_error("Mesh contains zero vertex normals");
    }

    // printf("Mesh #%d\n\tNumVertices: %d\n\tNumFaces: %d\n", m, mesh->mNumVertices, mesh->mNumFaces);

    // add points
    for (unsigned int i = 0u; i < mesh->mNumVertices; i++) {
      aiVector3D pos = mesh->mVertices[i];
      aiVector3D norm = mesh->mNormals[i];

      vertexMap[i + vertexOffset] = optix::make_float3(pos.x, pos.y, pos.z); // +aabb.center();
      normalMap[i + vertexOffset] = optix::normalize(optix::make_float3(norm.x, norm.y, norm.z));
    }

    // add faces
    for (unsigned int i = 0u; i < mesh->mNumFaces; i++) {

      aiFace face = mesh->mFaces[i];

      // add triangles
      if (face.mNumIndices == 3) {
        faceMap[i + faceOffset] = optix::make_uint3(
          face.mIndices[0] + vertexOffset, face.mIndices[1] + vertexOffset, face.mIndices[2] + vertexOffset);
      } else {
        // printf("face indices != 3\n");
        faceMap[i + faceOffset] = optix::make_uint3(-1);
      }
      materialsMap[i + faceOffset] = 0u;
    }

    // create geometry
    optix::Geometry geometry = m_context->createGeometry();

    geometry["vertex_buffer"]->setBuffer(m_vertices);
    geometry["normal_buffer"]->setBuffer(m_normals);
    geometry["index_buffer"]->setBuffer(m_faces);
    geometry["texcoord_buffer"]->setBuffer(m_tbuffer);
    geometry["material_buffer"]->setBuffer(m_materials);

    geometry->setPrimitiveCount(mesh->mNumFaces);
    geometry->setIntersectionProgram(programs.m_intersect);
    geometry->setBoundingBoxProgram(programs.m_boundingBox);
    geometry->setPrimitiveIndexOffset(faceOffset);

    optix::GeometryInstance gi = m_context->createGeometryInstance(geometry, &matl, &matl + 1);
    m_gis.push_back(gi);

    vertexOffset += mesh->mNumVertices;
    faceOffset += mesh->mNumFaces;
  }

  // add all geometry instances to a geometry group
  m_transform = m_context->createTransform();

  m_geometrygroup = m_context->createGeometryGroup();
  m_geometrygroup->setChildCount(static_cast<unsigned int>(m_gis.size()));
  for (unsigned i = 0u; i < m_gis.size(); i++) {
    m_geometrygroup->setChild(i, m_gis[i]);
  }
  optix::Acceleration a = m_context->createAcceleration("Trbvh");
  m_geometrygroup->setAcceleration(a);

  m_transform->setMatrix(false, glm::value_ptr(mtx), NULL);
  m_transform->setChild(m_geometrygroup);
}

void
OptiXMesh::destroy()
{}
