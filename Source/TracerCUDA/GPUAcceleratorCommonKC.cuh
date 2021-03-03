#pragma once

#include "RayLib/AABB.h"
#include "GPUTransformI.h"

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

__host__
static void TransformLocalAABBToWorld(AABB3f& aabb, const GPUTransformI& transform,
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