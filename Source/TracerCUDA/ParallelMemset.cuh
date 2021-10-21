#pragma once

#include "CudaSystem.h"

template <class T>
__global__
static void KCMemset(T* gData,
                     T hValue,
                     uint32_t count)
{
    // Kernel Grid - Stride Loop
    for(uint32_t globalId = threadIdx.x + blockDim.x * blockIdx.x;
        globalId < count;
        globalId += (blockDim.x * gridDim.x))
    {
        gData[globalId] = hValue;
    }
}