#pragma once

/**

GPU Linear Congruential Generator

Implementation of Warp Std Generator

*/

#include <cuda_runtime.h>
#include "RayLib/Vector.h"

class CudaGPU;

#define GLOBAL_ID_X (threadIdx.x + blockDim.x * blockIdx.x)
#define GLOBAL_ID_Y (threadIdx.y + blockDim.y * blockIdx.y)
#define LINEAR_GLOBAL_ID (GLOBAL_ID_X + GLOBAL_ID_Y * gridDim.x)

class RNGeneratorGPUI
{
    public:
    //virtual         ~RandomGPU() = default;
    // Interface
    __device__ __forceinline__
    virtual float       Uniform() = 0;
    __device__ __forceinline__
    virtual float       Uniform(float min, float max) = 0;
    __device__ __forceinline__
    virtual Vector2f    Uniform2D() = 0;
    __device__ __forceinline__
    virtual float       Normal() = 0;
    __device__ __forceinline__
    virtual float       Normal(float mean, float stdDev) = 0;
};

class RNGeneratorCPUI
{
    public:
    virtual                     ~RNGeneratorCPUI() = default;
    //
    virtual RNGeneratorGPUI**   GetGPUGenerators(const CudaGPU&) = 0;
    virtual size_t              UsedGPUMemory() const = 0;
};

namespace RNGAccessor
{
    template <class GPUGenerator>
    __device__
    GPUGenerator& Acquire(RNGeneratorGPUI**,
                          uint32_t index);
}

template <class GPUGenerator>
__device__ __forceinline__
GPUGenerator& RNGAccessor::Acquire(RNGeneratorGPUI** gGenerators,
                                   uint32_t index)
{
    return static_cast<GPUGenerator&>(*gGenerators[index]);
}


