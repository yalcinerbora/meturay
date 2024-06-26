#pragma once

#include "RayLib/HitStructs.h"
#include "RayStructs.h"
#include "RNGenerator.h"
#include "CudaSystem.hpp"

template <class T, class... Args>
__global__ void KCConstructGPUClass(T* gLocation,
                                    uint32_t classCount,
                                    Args... args)
{
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < classCount;
        globalId += blockDim.x * gridDim.x)
    {
        auto a = new (gLocation) T(args...);
    }
}

template<class RayAuxData>
using RayFinalizeFunc = void(*)(// Output
                                Vector4* gImage,
                                // Input
                                const RayAuxData&,
                                const RayReg&,
                                //
                                RNGeneratorGPUI& rng);

template <class RayAuxData, class RNG,
          RayFinalizeFunc<RayAuxData> FinalizeFunc >
__global__ void KCFinalizeRay(// Output
                              Vector4* gImage,
                              // Input
                              const RayGMem* gInRays,
                              const RayAuxData* gInRayAux,
                              //
                              const RayId* gRayIds,
                              //
                              const uint32_t rayCount,
                              RNGeneratorGPUI** gRNGs)
{
    // Pre-grid stride loop
    // RNG is allocated for each SM (not for each thread)
    auto& rng = RNGAccessor::Acquire<RNG>(gRNGs, LINEAR_GLOBAL_ID);

    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < rayCount; globalId += blockDim.x * gridDim.x)
    {
        const RayId rayId = gRayIds[globalId];

        // Load Input to Registers
        const RayReg ray(gInRays, rayId);
        const RayAuxData aux = gInRayAux[rayId];

        // Actual Shading
        FinalizeFunc(// Output
                     gImage,
                     // Input as registers
                     ray,
                     aux,
                     //
                     rng);
    }
}