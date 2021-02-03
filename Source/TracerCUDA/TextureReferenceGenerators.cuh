#pragma once

#include "Texture.cuh"
#include "TextureReference.cuh"
#include <cuda.h>

__global__
template <int D, class T>
void GenerateTextureReference(TextureRef<D, T>* gRefLocations,

                              //
                              const cudaTextureObject_t* gTextures,                              
                              
                              uint32_t refCount)

{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < refCount;
        globalId += blockDim.x * gridDim.x)
    {

        new (gRefLocations + globalId) TextureRef<D, T>(gTextures[globalId]);
    }
}

template <int D, class T>
void GenerateConstantReference(ConstantRef<D, T>* gRefLocations,
                               const T* gConstantData,
                               uint32_t refCount)

{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < refCount;
        globalId += blockDim.x * gridDim.x)
    {

        new (gRefLocations + globalId) ConstantRef<D, T>(gConstants[globalId]);
    }
}