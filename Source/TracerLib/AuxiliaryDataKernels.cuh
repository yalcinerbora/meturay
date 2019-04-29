#pragma once

#include "RayLib/HitStructs.h"
#include "RayStructs.h"
#include "Random.cuh"

template<class RayAuxData>
using RayFinalizeFunc = void(*)(// Output
                                Vector4* gImage,
                                // Input
                                const RayAuxData&,
                                const RayReg&,
                                //
                                RandomGPU& rng);


template <class RayAuxData, RayFinalizeFunc<RayAuxData> FinalizeFunc>
__global__ void KCFinalizeRay(// Output
                              Vector4* gImage,
                              // Input
                              const RayGMem* gInRays,
                              const RayAuxData* gInRayAux,
                              //
                              const RayId* gRayIds,
                              //
                              const uint32_t rayCount,
                              RNGGMem gRNGStates)
{
    // Pre-grid stride loop
    // RNG is allocated for each SM (not for each thread)
    extern __shared__ uint32_t sRNGStates[];
    RandomGPU rng(gRNGStates.state, sRNGStates);

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