#include "GPUPrimitiveSkySphere.h"

#include "RayLib/SceneError.h"
#include "RayLib/SceneNodeI.h"
#include "RayLib/SurfaceLoaderGenerator.h"
#include "RayLib/SurfaceLoaderI.h"
#include "RayLib/Log.h"

#include <array>

GPUPrimitiveSkySphere::GPUPrimitiveSkySphere()
    : totalPrimitiveCount(0)
{}

const char* GPUPrimitiveSkySphere::Type() const
{
    return TypeName();
}

SceneError GPUPrimitiveSkySphere::InitializeGroup(const NodeListing& surfaceDataNodes, double time,
                                                  const SurfaceLoaderGeneratorI& loaderGen,
                                                  const TextureNodeMap& textureNodes,
                                                  const std::string& scenePath)
{
    SceneError e = SceneError::OK;  
    size_t totalCount = 0;
    // First calculate size
    for(const auto& sPtr : surfaceDataNodes)
    {        
        size_t count = sPtr->IdCount();
        totalCount += count;     
    }

    // Reserve CPU Memory for loading
    std::vector<float> distancesCPU(totalCount);

    uint32_t globalIndex = 0;
    for(const auto& sPtr : surfaceDataNodes)
    {
        const SceneNodeI& s = *sPtr;

        OptionalNodeList<float> distLocal = s.AccessOptionalFloat(NAME_DISTANCE, time);

        uint32_t localIndex = 0;
        for(const auto& pair : s.Ids())
        {
            NodeId id = pair.first;

            float distance = distLocal[localIndex].first
                                ? distLocal[localIndex].second
                                : std::numeric_limits<float>::max();
            
            batchRanges.emplace(id, Vector2ul(globalIndex, globalIndex + 1));
            batchAABBs.emplace(id, AABB3f(Vector3f(-distance), 
                                          Vector3f(distance)));

            distancesCPU[globalIndex] = distance;
            
            localIndex++;
        }
        globalIndex++;
    }

    // All loaded into CPU Memory now load it into GPU memory
    float* dDistances;
    DeviceMemory::AllocateMultiData(std::tie(dDistances), memory, {totalCount});

    // Copy Data to GPU
    CUDA_CHECK(cudaMemcpy(dDistances, distancesCPU.data(),
                          sizeof(float) * totalCount,
                          cudaMemcpyHostToDevice));

    dData.radius = dDistances;

    // All Done!
    return SceneError::OK;
}

SceneError GPUPrimitiveSkySphere::ChangeTime(const NodeListing& surfaceDatalNodes, double time,
                                             const SurfaceLoaderGeneratorI& loaderGen,
                                             const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR;    
}

Vector2ul GPUPrimitiveSkySphere::PrimitiveBatchRange(uint32_t surfaceDataId) const
{
    return batchRanges.at(surfaceDataId);
    
}

AABB3 GPUPrimitiveSkySphere::PrimitiveBatchAABB(uint32_t surfaceDataId) const
{
    return batchAABBs.at(surfaceDataId);
}

PrimTransformType GPUPrimitiveSkySphere::TransformType() const
{
    return PrimTransformType::CONSTANT_LOCAL_TRANSFORM;
}

bool GPUPrimitiveSkySphere::CanGenerateData(const std::string& s) const
{
    return false;
}