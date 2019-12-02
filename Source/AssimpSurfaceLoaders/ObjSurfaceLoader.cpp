#include "ObjSurfaceLoader.h"

#include "RayLib/SceneError.h"
#include "RayLib/SceneNodeI.h"

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

ObjSurfaceLoader::ObjSurfaceLoader(Assimp::Importer& i,
                                   const std::string& scenePath,
                                   const SceneNodeI&, double time)
: SurfaceLoader(node, time)
, importer(i)
, scene(nullptr)
{
    // Get File Name
    const std::string name = node.Name();
    // Prepend

    scene = importer.ReadFile(name,
                              aiProcess_CalcTangentSpace |
                              aiProcess_GenNormals |
                              aiProcess_Triangulate |
                              aiProcess_JoinIdenticalVertices |
                              aiProcess_SortByPType);

    // Report Failed Import
    if(!scene) throw SceneException(SceneError::SURFACE_DATA_INVALID_READ,
                                    importer.GetErrorString());

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
    return SceneError::OK;
}

SceneError ObjSurfaceLoader::PrimitiveRanges(std::vector<Vector2ul>&) const
{
    return SceneError::OK;
}

SceneError ObjSurfaceLoader::PrimitiveCounts(std::vector<size_t>&) const
{
    return SceneError::OK;
}

SceneError ObjSurfaceLoader::PrimitiveDataRanges(std::vector<Vector2ul>&) const
{
    return SceneError::OK;
}

SceneError ObjSurfaceLoader::GetPrimitiveData(Byte*, PrimitiveDataType primitiveDataType) const
{
    return SceneError::OK;
}

SceneError ObjSurfaceLoader::PrimitiveDataCount(size_t&, PrimitiveDataType primitiveDataType) const
{
    return SceneError::OK;
}

SceneError ObjSurfaceLoader::PrimDataLayout(PrimitiveDataLayout&,
                                            PrimitiveDataType primitiveDataType) const
{
    return SceneError::OK;
}