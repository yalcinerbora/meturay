#pragma once

#include "RayLib/HitStructs.h"
#include "RayStructs.h"
#include "Random.cuh"
#include "ImageStructs.h"

template <class MGroup, class PGroup>
using SurfaceFunc = MGroup::Surface(*)(const PGroup::PrimitiveData&,
                                       const PGroup::HitData&,
                                       PrimitiveId);
//
template <class TLogic, class ELogic, class Surface, class MaterialData>
using ShadeFunc = void(*)(// Output
                          ImageGMem<Vector4f> gImage,
                          //
                          HitKey* gOutBoundKeys,
                          RayGMem* gOutRays,
                          TLogic::RayAuxData* gOutRayAux,
                          const uint32_t maxOutRay,
                          // Input as registers
                          const RayReg& ray,
                          const Surface& surface,
                          const TLogic::RayAuxData& aux,
                          // RNG
                          RandomGPU& rng,
                          // Estimator
                          const ELogic::EstimatorData& estData,
                          // Input as global memory
                          const MaterialData& gMatData,
                          const HitKey::Type& matId);

template <class TLogic, class ELogic, class MGroup, class PGroup,
          SurfaceFunc<MGroup, PGroup> SurfFunc>
__global__ void KCMaterialShade(// Output
                                ImageGMem<Vector4f> gImage,
                                //
                                HitKey* gOutBoundKeys,
                                RayGMem* gOutRays,
                                TLogic::RayAuxData* gOutRayAux,
                                const uint32_t maxOutRay,
                                // Input
                                const RayGMem* gInRays,
                                const TLogic::RayAuxData* gInRayAux,
                                const PrimitiveId* gPrimitiveIds,
                                const HitStructPtr gHitStructs,
                                //
                                const HitKey* gMatIds,
                                const RayId* gRayIds,
                                //
                                const uint32_t rayCount,
                                RNGGMem gRNGStates,
                                // Estimator
                                const ELogic::EstimatorData est,
                                // Material Related
                                const MGroup::MaterialData matData,
                                // Primitive Related
                                const PGroup::PrimitiveData primData)
{
    // Fetch Types from Template Classes
    using HitData = typename PGroup::HitData;               // HitData is defined by primitive
    using Surface = typename MGroup::Surface;               // Surface is defined by material group
    using RayAuxData = typename TLogic::RayAuxData;         // Hit register defined by primitive

    // Pre-grid stride loop
    // RNG is allocated for each SM (not for each thread)
    RandomGPU rng(gRNGStates.state);

    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < rayCount; globalId += blockDim.x * gridDim.x)
    {
        const RayId rayId = gRayIds[globalId];
        const HitKey hitKey = gMatIds[globalId];

        // Load Input to Registers
        const RayReg ray(gInRays, rayId);
        const RayAuxData aux = gInRayAux[rayId];
        const PrimitiveId gPrimitiveId = gPrimitiveIds[rayId];

        // Generate surface data from hit
        const HitData hit = gHitStructs.Ref<HitData>(rayId);
        const Surface surface = SurfFunc(primData, hit,
                                         gPrimitiveId);

        // Determine Output Location
        RayGMem* gLocalRayOut = gOutRays + globalId * maxOutRay;
        RayAuxData* gLocalAuxOut = gOutRayAux + globalId * maxOutRay;
        HitKey* gLocalBoundKeyOut = gOutBoundKeys + globalId * maxOutRay;
        // Actual Shading
        MGroup::ShadeFunc(// Output
                          gImage,
                          //
                          gLocalBoundKeyOut,
                          gLocalRayOut,
                          gLocalAuxOut,
                          maxOutRay,
                          // Input as registers
                          ray,
                          surface,
                          aux,
                          // RNG
                          rng,
                          // Estimator
                          est,
                          // Input as global memory
                          matData,
                          HitKey::FetchIdPortion(hitKey));
    }
}