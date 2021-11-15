#pragma once

#include "RayLib/AABB.h"
#include "GPUTransformI.h"

__global__
static void KCInitIndices(// O
                          uint32_t* gIndices,
                          PrimitiveId* gPrimIds,
                          // Input
                          uint32_t indexStart,
                          uint64_t rangeStart,
                          uint32_t primCount)
{
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < primCount; globalId += blockDim.x * gridDim.x)
    {
        gIndices[globalId] = indexStart + globalId;
        gPrimIds[globalId] = rangeStart + globalId;
    }
}

__global__
static void KCTransformAABB(AABB3f& gAABB,
                            const GPUTransformI& transform,
                            const AABB3f localAABB)
{
    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalId == 0)
    {
        gAABB = transform.LocalToWorld(localAABB);
    }
}

__global__
static void KCAcquireTransform(const GPUTransformI** gTransformPtr,
                               const uint32_t* gAccTransformIdList,
                               const GPUTransformI** gTransformList,
                               uint32_t innerIndex)
{
    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalId == 0)
    {
        *gTransformPtr = gTransformList[gAccTransformIdList[innerIndex]];
    }
}

__global__
static void KCAcquireTransform(const GPUTransformI** gTransformPtr,
                               const GPUTransformI** gTransformList,
                               uint32_t index)
{
    uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalId == 0)
    {
        *gTransformPtr = gTransformList[index];
    }
}

__host__
inline void TransformLocalAABBToWorld(AABB3f& aabb, const GPUTransformI& transform,
                                      const CudaGPU& gpu)
{
    DeviceMemory tempMem(sizeof(AABB3f));
    // Generate World Space AABB from Local AABB
    gpu.KC_X(0, 0, 1,
             KCTransformAABB,
             *static_cast<AABB3f*>(tempMem),
             transform,
             aabb);

    AABB3f aabbReturn;
    CUDA_CHECK(cudaMemcpy(&aabbReturn, tempMem, sizeof(AABB3f),
                          cudaMemcpyDeviceToHost));
    aabb = aabbReturn;
}

__host__
inline void AcquireAcceleratorGPUTransform(const GPUTransformI*& transform,
                                           const uint32_t* dAccTransformIdList,
                                           const GPUTransformI** dTransformList,
                                           uint32_t innerIndex,
                                           const CudaGPU& gpu)
{
    DeviceMemory tempMem(sizeof(GPUTransformI*));
    // Generate World Space AABB from Local AABB
    gpu.KC_X(0, 0, 1,
             static_cast<void (*)(const GPUTransformI**, const uint32_t*,
                                  const GPUTransformI**, uint32_t)>
             (KCAcquireTransform),
             static_cast<const GPUTransformI**>(tempMem),
             dAccTransformIdList,
             dTransformList,
             innerIndex);

    GPUTransformI* ptr;
    CUDA_CHECK(cudaMemcpy(&ptr, tempMem, sizeof(GPUTransformI*),
                          cudaMemcpyDeviceToHost));
    transform = ptr;
}

__host__
inline void AcquireIdentityTransform(const GPUTransformI*& transform,
                                     const GPUTransformI** dTransformList,
                                     uint32_t identityTransformIndex,
                                     const CudaGPU& gpu)
{
    DeviceMemory tempMem(sizeof(GPUTransformI*));
    // Generate World Space AABB from Local AABB
    gpu.KC_X(0, 0, 1,
             static_cast<void (*)(const GPUTransformI**, const GPUTransformI**,
                                  uint32_t)>
             (KCAcquireTransform),
             static_cast<const GPUTransformI**>(tempMem),
             dTransformList,
             identityTransformIndex);

    GPUTransformI* ptr;
    CUDA_CHECK(cudaMemcpy(&ptr, tempMem, sizeof(GPUTransformI*),
                          cudaMemcpyDeviceToHost));
    transform = ptr;
}

__global__
static void KCInitPrimIdsAndIndices(// O
                                    uint32_t* gIndices,
                                    PrimitiveId* gPrimIds,
                                    // Input
                                    uint32_t indexStart,
                                    uint64_t rangeStart,
                                    uint32_t primCount)
{
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < primCount; globalId += blockDim.x * gridDim.x)
    {
        gIndices[globalId] = indexStart + globalId;
        gPrimIds[globalId] = rangeStart + globalId;
    }
}

template <class PGroup>
__global__
static void KCGenCentersWithIndex(// O
                                  Vector3f* gCenters,
                                  // Input
                                  const uint32_t* gIndicies,
                                  const PrimitiveId* gPrimitiveIds,
                                  //
                                  CentroidGen<PGroup> centFunc,
                                  uint32_t primCount)
{
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < primCount; globalId += blockDim.x * gridDim.x)
    {
        uint32_t id = gIndicies[globalId];
        PrimitiveId primId = gPrimitiveIds[id];

        gCenters[globalId] = centFunc(primId);
    }
}

template <class PGroup>
__global__
static void KCGenAABBsWithIndex(// O
                                AABB3f* gAABBs,
                                // Input
                                const uint32_t* gIndicies,
                                const PrimitiveId* gPrimitiveIds,
                                //
                                AABBGen<PGroup> aabbFunc,
                                uint32_t primCount)
{
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < primCount; globalId += blockDim.x * gridDim.x)
    {
        uint32_t id = gIndicies[globalId];
        PrimitiveId primId = gPrimitiveIds[id];

        gAABBs[globalId] = aabbFunc(primId);
    }
}

template <class PGroup>
__global__
static void KCGenAABBs(// O
                       AABB3f* gAABBs,
                       // Input
                       Vector2ul primRange,
                       //
                       AABBGen<PGroup> aabbFunc,
                       uint32_t primCount)
{
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < primCount; globalId += blockDim.x * gridDim.x)
    {
        PrimitiveId primId = primRange[0] + globalId;
        gAABBs[globalId] = aabbFunc(primId);
    }
}
