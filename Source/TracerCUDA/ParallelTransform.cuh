#pragma once

#include "RayLib/CudaCheck.h"
#include "TransformFunctions.cuh"

template <class Type, class ReduceFunctor>
__global__ void TransformArray(Type* gInOut,
                               unsigned int elementCount,
                               ReduceFunctor f)
{
    unsigned int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalId >= elementCount) return;

    Type input = gInOut[globalId];
    Type result = f(input);
    //gIn[globalId] = result;
}

template<class Type, class ReduceFunctor>
__host__ void TransformArrayGPU(Type* dData,
                                size_t elementCount,
                                const ReduceFunctor& f,
                                cudaStream_t stream = (cudaStream_t)0)
{
    static constexpr unsigned int TPB = StaticThreadPerBlock1D;
    const unsigned int gridSize = static_cast<uint32_t>((elementCount + TPB - 1) / TPB);
    TransformArray<<<gridSize, TPB, 0, stream>>>
    (
        dData,
        static_cast<uint32_t>(elementCount),
        f
    );
}