#include "ObjSurfaceLoader.h"

#include "RayLib/SceneError.h"
#include "RayLib/SceneNodeI.h"
#include "RayLib/PrimitiveDataTypes.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include "RayLib/FileSystemUtility.h"

ObjSurfaceLoader::ObjSurfaceLoader(Assimp::Importer& i,
                                   const std::string& scenePath,
                                   const SceneNodeI& node, double time)
: SurfaceLoader(node, time)
, importer(i)
, scene(nullptr)
{    
    // Get File Name
    const std::string filePath = Utilitiy::MergeFileFolder(scenePath, node.Name());

    scene = importer.ReadFile(filePath,
                              aiProcess_CalcTangentSpace |
                              aiProcess_GenBoundingBoxes |
                              aiProcess_GenNormals |
                              aiProcess_Triangulate |
                              aiProcess_JoinIdenticalVertices |
                              aiProcess_SortByPType |
                              aiProcess_RemoveRedundantMaterials);

    // Report Failed Import
    if(!scene) throw SceneException(SceneError::SURFACE_DATA_INVALID_READ,
                                    importer.GetErrorString());

    // Do some checking
    const auto& innerIds = node.CommonUIntList(InnerIdJSON);
    for(unsigned int innerId : innerIds)
    {
        const auto& mesh = scene->mMeshes[innerId];

        if(innerId >= scene->mNumMeshes)
            throw SceneException(SceneError::SURFACE_LOADER_INTERNAL_ERROR,
                                 "Assimp_obj: Inner index out of range");
        if(!mesh->HasNormals())
            throw SceneException(SceneError::SURFACE_LOADER_INTERNAL_ERROR,
                                 "Assimp_obj: Obj file does not have normals");
        if(!mesh->HasPositions())
            throw SceneException(SceneError::SURFACE_LOADER_INTERNAL_ERROR,
                                 "Assimp_obj: Obj file does not have positions");
        if(!mesh->GetNumUVChannels())
            throw SceneException(SceneError::SURFACE_LOADER_INTERNAL_ERROR,
                                 "Assimp_obj: Obj file does not have uvs");
        if(!(mesh->mPrimitiveTypes == aiPrimitiveType_TRIANGLE))
            throw SceneException(SceneError::SURFACE_LOADER_INTERNAL_ERROR,
                                 "Assimp_obj: Only triangle is supported");
    }
    // Take owneship of the scene
    scene = importer.GetOrphanedScene();
}

ObjSurfaceLoader::~ObjSurfaceLoader()
{
    if(scene) delete scene;
}

const char* ObjSurfaceLoader::SufaceDataFileExt() const
{
    return Extension();
}

SceneError ObjSurfaceLoader::AABB(std::vector<AABB3>& list) const
{
    const auto& innerIds = node.CommonUIntList(InnerIdJSON);
    for(unsigned int innerId : innerIds)
    {
        const auto& aabb = scene->mMeshes[innerId]->mAABB;
        list.push_back(AABB3(Vector3(aabb.mMin.x,
                                     aabb.mMin.y,
                                     aabb.mMin.z),
                             Vector3(aabb.mMax.x,
                                     aabb.mMax.y,
                                     aabb.mMax.z)));
    }
    return SceneError::OK;
}

SceneError ObjSurfaceLoader::PrimitiveRanges(std::vector<Vector2ul>& result) const
{
    // Self contain indices
    size_t prevOffset = 0;
    const auto& innerIds = node.CommonUIntList(InnerIdJSON);
    for(unsigned int innerId : innerIds)
    {
        const auto& mesh = scene->mMeshes[innerId];
        result.emplace_back(prevOffset, 0);
        size_t meshIndexCount = 0;

        for(unsigned int i = 0; i < mesh->mNumFaces; i++)
        {
            const auto& face = mesh->mFaces[i];
            meshIndexCount += face.mNumIndices;
        }
        if(meshIndexCount % 3 != 0) return SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR;
        meshIndexCount /= 3;

        result.back()[1] = result.back()[0] + meshIndexCount;        
        prevOffset += meshIndexCount;
    }
    return SceneError::OK;
}

SceneError ObjSurfaceLoader::PrimitiveCounts(std::vector<size_t>& result) const
{
    // Self contain indices
    const auto& innerIds = node.CommonUIntList(InnerIdJSON);
    for(unsigned int innerId : innerIds)
    {
        const auto& mesh = scene->mMeshes[innerId];
        result.emplace_back(0);
        for(unsigned int i = 0; i < mesh->mNumFaces; i++)
        {
            const auto& face = mesh->mFaces[i];
            result.back() += face.mNumIndices;
        }
    }
    return SceneError::OK;
}

SceneError ObjSurfaceLoader::PrimitiveDataRanges(std::vector<Vector2ul>& result) const
{
    // Self contain indices
    size_t prevOffset = 0;
    const auto& innerIds = node.CommonUIntList(InnerIdJSON);
    for(unsigned int innerId : innerIds)
    {
        const auto& mesh = scene->mMeshes[innerId];
        result.emplace_back(prevOffset, 0);
        size_t vertexCount = mesh->mNumVertices;
        result.back()[1] = result.back()[0] + vertexCount;
        prevOffset += vertexCount;
    }
    return SceneError::OK;
}

SceneError ObjSurfaceLoader::GetPrimitiveData(Byte* result, PrimitiveDataType primitiveDataType) const
{
    // Self contain indices
    Byte* meshStart = result;
    const auto& innerIds = node.CommonUIntList(InnerIdJSON);
    for(unsigned int innerId : innerIds)
    {
        const auto& mesh = scene->mMeshes[innerId];
        switch(primitiveDataType)
        {
            case PrimitiveDataType::POSITION:
            {
                std::memcpy(meshStart, mesh->mVertices,
                            sizeof(Vector3) * mesh->mNumVertices);
                meshStart += sizeof(Vector3) * mesh->mNumVertices;
                break;
            }
            case PrimitiveDataType::NORMAL:
            {
                std::memcpy(meshStart, mesh->mNormals,
                            sizeof(Vector3) * mesh->mNumVertices);
                meshStart += sizeof(Vector3) * mesh->mNumVertices;
                break;
            }
            case PrimitiveDataType::UV:
            {
                // Do manual copy since data is strided
                Vector2* uvStart = reinterpret_cast<Vector2*>(meshStart);
                for(unsigned int i = 0; i < mesh->mNumVertices; i++)
                {
                    uvStart[i][0] = mesh->mTextureCoords[0][i].x;
                    uvStart[i][1] = mesh->mTextureCoords[0][i].y;
                }
                meshStart += sizeof(Vector2) * mesh->mNumVertices;
                break;
            }
            case PrimitiveDataType::VERTEX_INDEX:
            {
                uint64_t* indexStart = reinterpret_cast<uint64_t*>(meshStart);                                
                for(unsigned int i = 0; i < mesh->mNumFaces; i++)
                {
                    const auto& face = mesh->mFaces[i];
                    indexStart[i * 3 + 0] = face.mIndices[0];
                    indexStart[i * 3 + 1] = face.mIndices[1];
                    indexStart[i * 3 + 2] = face.mIndices[2];
                }
                meshStart += sizeof(uint64_t) * 3 * mesh->mNumFaces;
                break;
            }
            default:
                return SceneError::SURFACE_DATA_TYPE_NOT_FOUND;
        }
    }
    return SceneError::OK;
}

SceneError ObjSurfaceLoader::PrimitiveDataCount(size_t& total, PrimitiveDataType primitiveDataType) const
{
    total = 0;
    const auto& innerIds = node.CommonUIntList(InnerIdJSON);
    for(unsigned int innerId : innerIds)
    {
        const auto& mesh = scene->mMeshes[innerId];
        switch(primitiveDataType)
        {
            case PrimitiveDataType::POSITION:
            case PrimitiveDataType::NORMAL:
            case PrimitiveDataType::UV:
            {
                total += mesh->mNumVertices;
                break;
            }
            case PrimitiveDataType::VERTEX_INDEX:
            {
                for(unsigned int i = 0; i < mesh->mNumFaces; i++)
                {
                    const auto& face = mesh->mFaces[i];
                    total += face.mNumIndices;
                }
                break;
            }
        }
    }
    return SceneError::OK;
}

SceneError ObjSurfaceLoader::PrimDataLayout(PrimitiveDataLayout& result,
                                            PrimitiveDataType primitiveDataType) const
{

    if(primitiveDataType == PrimitiveDataType::POSITION ||
       primitiveDataType == PrimitiveDataType::NORMAL)
        result = PrimitiveDataLayout::FLOAT_3;
    else if(primitiveDataType == PrimitiveDataType::UV)
        result = PrimitiveDataLayout::FLOAT_2;
    else if(primitiveDataType == PrimitiveDataType::VERTEX_INDEX)
        result = PrimitiveDataLayout::UINT64_1;
    else return SceneError::SURFACE_DATA_TYPE_NOT_FOUND;
    return SceneError::OK;
}