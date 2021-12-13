#include "RayLib/CudaCheck.h"
#include "DeviceMemory.h"
#pragma once

#include "CudaSystem.h"
#include "ReduceFunctions.cuh"

template <class Type>
__global__ void KCIota(Type* gOut,
                       Type startingElement,
                       uint32_t elementCount)
{
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < elementCount; globalId += blockDim.x * gridDim.x)
    {
        gOut[globalId] = startingElement + globalId;
    }
}

template <class Type>
__global__ void KCExpand(Type* gOut,
                         Type* gExpandingValue,
                         uint32_t elementCount)
{
    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < elementCount; globalId += blockDim.x * gridDim.x)
    {
        // TODO: Save this to a shared memory for each block etc..
        gOut[globalId] = *gExpandingValue;
    }
}

template<class Type>
__host__ void IotaGPU(Type* dArray, Type startingElement,
                      size_t count, cudaStream_t stream = (cudaStream_t)0)
{
    static constexpr unsigned int TPB = StaticThreadPerBlock1D;
    unsigned int gridSize = static_cast<unsigned int>((count + TPB - 1) / TPB);

    // KC Parallel Reduction
    KCIota<Type><<<gridSize, TPB, 0, stream>>>
    (
        dArray,
        startingElement,
        static_cast<uint32_t>(count)
    );
    CUDA_KERNEL_CHECK();
}


template<class Type>
__host__ void ExpandValueGPU(Type* dArray, Type* dExpandingValue,
                             size_t count, cudaStream_t stream = (cudaStream_t)0)
{
    static constexpr unsigned int TPB = StaticThreadPerBlock1D;
    unsigned int gridSize = static_cast<unsigned int>((count + TPB - 1) / TPB);

    // KC Parallel Reduction
    KCExpand<Type><<<gridSize, TPB, 0, stream>>>
    (
        dArray,
        dExpandingValue,
        static_cast<uint32_t>(count)
    );
    CUDA_KERNEL_CHECK();
}