#include "GPUPrimitiveDirectional.h"

#include "RayLib/SceneError.h"
#include "RayLib/SceneNodeI.h"
#include "RayLib/SurfaceLoaderGenerator.h"
#include "RayLib/SurfaceLoaderI.h"
#include "RayLib/Log.h"

#include <array>

GPUPrimitiveDirectional::GPUPrimitiveDirectional()
    : totalPrimitiveCount(0)
{}

const char* GPUPrimitiveDirectional::Type() const
{
    return TypeName();
}

SceneError GPUPrimitiveDirectional::InitializeGroup(const NodeListing& surfaceDataNodes, double time,
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
    std::vector<Vector3f> directionsCPU(totalCount);
    std::vector<AABB3f> aabbsCPU(totalCount);
    std::vector<float> distancesCPU(totalCount);

    uint32_t globalIndex = 0;
    for(const auto& sPtr : surfaceDataNodes)
    {
        const SceneNodeI& s = *sPtr;

        std::vector<Vector3f> aabbMin = s.AccessVector3(NAME_SPAN_MIN, time);
        std::vector<Vector3f> aabbMax = s.AccessVector3(NAME_SPAN_MAX, time);
        std::vector<Vector3f> dirLocal = s.AccessVector3(NAME_DIRECTION, time);
        OptionalNodeList<float> distLocal = s.AccessOptionalFloat(NAME_DISTANCE, time);

        uint32_t localIndex = 0;
        for(const auto& pair : s.Ids())
        {
            NodeId id = pair.first;

            AABB3f aabb = AABB3f(aabbMin[localIndex],
                                 aabbMax[localIndex]);

            batchRanges.emplace(id, Vector2ul(globalIndex, globalIndex + 1));
            batchAABBs.emplace(id, aabb);

            aabbsCPU[globalIndex] = aabb;
            directionsCPU[globalIndex] = dirLocal[localIndex];
            distancesCPU[globalIndex] = distLocal[localIndex].first 
                                        ? distLocal[localIndex].second
                                        : std::numeric_limits<float>::max();
            localIndex++;
        }
        globalIndex++;
    }

    // All loaded into CPU Memory now load it into GPU memory
    Vector3f* dDirections;
    AABB3f* dAABBs;
    float* dDistances;
    DeviceMemory::AllocateMultiData(std::tie(dDirections, dAABBs, dDistances),
                                    memory, {totalCount, totalCount, totalCount});

    // Copy Data to GPU
    CUDA_CHECK(cudaMemcpy(dDirections, directionsCPU.data(),
                          sizeof(Vector3f) * totalCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dAABBs, aabbsCPU.data(),
                          sizeof(AABB3f) * totalCount,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dDistances, distancesCPU.data(),
                          sizeof(float) * totalCount,
                          cudaMemcpyHostToDevice));

    dData.directions = dDirections;
    dData.distances = dDistances;
    dData.spans = dAABBs;

    // All Done!
    return SceneError::OK;
}

SceneError GPUPrimitiveDirectional::ChangeTime(const NodeListing& surfaceDatalNodes, double time,
                                               const SurfaceLoaderGeneratorI& loaderGen,
                                               const std::string& scenePath)
{
    // TODO: Implement
    return SceneError::PRIMITIVE_TYPE_INTERNAL_ERROR;    
}

Vector2ul GPUPrimitiveDirectional::PrimitiveBatchRange(uint32_t surfaceDataId) const
{
    return batchRanges.at(surfaceDataId);
    
}

AABB3 GPUPrimitiveDirectional::PrimitiveBatchAABB(uint32_t surfaceDataId) const
{
    return batchAABBs.at(surfaceDataId);
}

PrimTransformType GPUPrimitiveDirectional::TransformType() const
{
    return PrimTransformType::CONSTANT_LOCAL_TRANSFORM;
}

bool GPUPrimitiveDirectional::IsIntersectable() const
{
    return false;
}

bool GPUPrimitiveDirectional::CanGenerateData(const std::string& s) const
{
    return false;
}