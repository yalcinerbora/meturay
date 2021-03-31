#pragma once

#include "Texture.cuh"
#include "TextureReference.cuh"
#include <cuda.h>

template <class T>
struct TextureOrConstReferenceData
{
    bool isConstData;
    cudaTextureObject_t tex;
    T data;
};

template <int D, class T>
__global__
void GenerateTextureReference(TextureRef<D, T>* gRefLocations,
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
__global__
void GenerateConstantReference(ConstantRef<D, T>* gRefLocations,
                               const T* gConstantData,
                               uint32_t refCount)

{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < refCount;
        globalId += blockDim.x * gridDim.x)
    {
        new (gRefLocations + globalId) ConstantRef<D, T>(gConstantData[globalId]);
    }
}

template <int D, class T>
__global__
void GenerateEitherTexOrConstantReference(TextureRefI<D, T>** gTexRefInterfaces,
                                          ConstantRef<D, T>* gCRefLocations,
                                          TextureRef<D, T>* gTRefLocations,
                                          // Atomic Counters
                                          uint32_t& gTRefCounter,
                                          uint32_t& gCRefCounter,
                                          //
                                          const TextureOrConstReferenceData<T>* gRefData,
                                          uint32_t totalRefCount)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < totalRefCount;
        globalId += blockDim.x * gridDim.x)
    {
        const TextureOrConstReferenceData<T> data = gRefData[globalId];

        TextureRefI<D, T>* refAddress = nullptr;
        if(data.isConstData)
        {
            //printf("Generating ConstantReference: %f, %f, %f\n",
            //       data.data[0], data.data[1], data.data[2]);
            uint32_t location = atomicAdd(&gCRefCounter, 1);
            refAddress = new (gCRefLocations + location) ConstantRef<D, T>(data.data);
        }
        else
        {
            uint32_t location = atomicAdd(&gTRefCounter, 1);
            //printf("Generating TexReference: %llu\n", data.tex);
            refAddress = new (gTRefLocations + location) TextureRef<D, T>(data.tex);
        }

        gTexRefInterfaces[globalId] = refAddress;
    }
}

template <int D, class T>
__global__
void GenerateOptionalTexReference(TextureRefI<D, T>** gTexRefInterfaces,
                                  TextureRef<D, T>* gTRefLocations,
                                  // Atomic Counter
                                  uint32_t& gTRefCounter,
                                  //
                                  const cudaTextureObject_t* gTexObjects,
                                  uint32_t totalRefCount)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < totalRefCount;
        globalId += blockDim.x * gridDim.x)
    {
        const cudaTextureObject_t tObj = gTexObjects[globalId];
        TextureRefI<D, T>* refAddress = nullptr;
        if(tObj != 0)
        {
            uint32_t location = atomicAdd(&gTRefCounter, 1);
            refAddress = new (gTRefLocations + location) TextureRef<D, T>(tObj);
        }
        gTexRefInterfaces[globalId] = refAddress;
    }
}

//template <int D, class T>
//__global__
//void GenerateEitherTexOrConstantReference(TextureRefI<D, T>** gTexRefInterfaces,
//                                          ConstantRef<D, T>* gCRefLocations,
//                                          TexArrayRef<D, T>* gTRefLocations,
//                                          // Atomic Counters
//                                          uint32_t& gTRefCounter,
//                                          uint32_t& gCRefCounter,
//                                          //
//                                          const TextureOrConstReferenceData<T>* gRefData,
//                                          uint32_t totalRefCount)
//{
//    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
//        globalId < totalRefCount;
//        globalId += blockDim.x * gridDim.x)
//    {
//        const ReferenceDataList<T> data = gRefData[globalId];
//
//        TextureRefI<D, T>* refAddress = nullptr;
//        if(data.isConstData)
//        {
//            uint32_t location = atomicAdd(&gCRefCounter, 1);
//            refAddress = new (gCRefLocations + location) ConstantRef<D, T>(data.data);
//        }
//        else
//        {
//            uint32_t location = atomicAdd(&gTRefCounter, 1);
//            refAddress = new (gTRefLocations + location) TexArrayRef<D, T>(data.tex,
//                                                                           data.textureArrayIndex);
//        }
//        gTexRefInterfaces[globalId] = refAddress;
//    }
//}