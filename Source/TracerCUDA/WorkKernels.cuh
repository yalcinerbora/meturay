#pragma once

#include "RayLib/HitStructs.h"
#include "RayStructs.h"
#include "Random.cuh"
#include "ImageStructs.h"
#include "GPUPrimitiveP.cuh"
#include "CudaSystem.hpp"

// Device Work Function Template
//
// Renderer supplies these functions with respect to material groups
// Renderer can define multiples of such functions for a same material
// and delegates to the tracer for each iteration.
//
// Each work function can generate multiples of rays.
// there is a static output limit which is determined by material and the
// work function itself.
//
// Work function may evaluate the MGroup or it may do some global work.
// It can also do both.
template <class GlobalState, class LocalState,
          class RayAuxiliary, class MGroup>
using WorkFunc = void(*)(// Output
                         HitKey* gOutBoundKeys,
                         RayGMem* gOutRays,
                         RayAuxiliary* gOutRayAux,
                         const uint32_t maxOutRay,
                         // Input as registers
                         const RayReg& ray,
                         const RayAuxiliary& aux,
                         const typename MGroup::Surface& surface,
                         const RayId rayId,
                         // I-O
                         LocalState& localState,
                         GlobalState& renderState,
                         RandomGPU& rng,
                         // Constants
                         const typename MGroup::Data& gMatData,
                         const HitKey::Type matIndex);

// Boundary Work Function is Slightly Different
// It also provides the actual enpoint that is being hit
template <class GlobalState, class LocalState,
          class RayAuxiliary, class EGroup>
using BoundaryWorkFunc = void(*)(// Output
                                 HitKey* gOutBoundKeys,
                                 RayGMem* gOutRays,
                                 RayAuxiliary* gOutRayAux,
                                 const uint32_t maxOutRay,
                                 // Input as registers
                                 const RayReg& ray,
                                 const RayAuxiliary& aux,
                                 const typename EGroup::Surface& surface,
                                 const RayId rayId,
                                 // I-O
                                 LocalState& localState,
                                 GlobalState& renderState,
                                 RandomGPU& rng,
                                 // Constants
                                 const typename EGroup::GPUType& gEndpoint);

// Meta Kernel for divding work.
template<class GlobalState, class LocalState,
         class RayAuxiliary, class PGroup, class MGroup,
         WorkFunc<GlobalState, LocalState, RayAuxiliary, MGroup> WFunc,
         SurfaceFunc<typename MGroup::Surface,
                     typename PGroup::HitData,
                     typename PGroup::PrimitiveData> SurfFunc>
__global__ CUDA_LAUNCH_BOUNDS_1D
void KCWork(// Output
            HitKey* gOutBoundKeys,
            RayGMem* gOutRays,
            RayAuxiliary* gOutRayAux,
            const uint32_t maxOutRay,
            // Input
            const RayGMem* gInRays,
            const RayAuxiliary* gInRayAux,
            const PrimitiveId* gPrimitiveIds,
            const TransformId* gTransformIds,
            const HitStructPtr gHitStructs,
            //
            const HitKey* gMatIds,
            const RayId* gRayIds,
            // I-O
            LocalState localState,
            GlobalState renderState,
            RNGGMem gRNGStates,
            // Constants
            const uint32_t rayCount,
            const typename MGroup::Data matData,
            const typename PGroup::PrimitiveData primData,
            const GPUTransformI* const* gTransforms)
{
    // Fetch Types from Template Classes
    using HitData = typename PGroup::HitData;   // HitData is defined by primitive
    using Surface = typename MGroup::Surface;   // Surface is defined by material group

    // Pre-grid stride initialization
    // RNG is allocated for each SM (not for each ray)
    RandomGPU rng(gRNGStates, LINEAR_GLOBAL_ID);

    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < rayCount; globalId += blockDim.x * gridDim.x)
    {
        const RayId rayId = gRayIds[globalId];
        const HitKey hitKey = gMatIds[globalId];

        // Load Input to Registers
        const RayReg ray(gInRays, rayId);
        const RayAuxiliary aux = gInRayAux[rayId];
        const PrimitiveId primitiveId = gPrimitiveIds[rayId];
        const TransformId transformId = gTransformIds[rayId];

        // Acquire transform for surface generation
        const GPUTransformI& transform = *gTransforms[transformId];

        // Generate surface data from hit
        const HitData hit = gHitStructs.Ref<HitData>(rayId);
        const Surface surface = SurfFunc(hit, transform, primitiveId, primData);

        // Determine Output Location
        // Make it locally indexable
        RayGMem* gLocalRayOut = gOutRays + globalId * maxOutRay;
        RayAuxiliary* gLocalAuxOut = gOutRayAux + globalId * maxOutRay;
        HitKey* gLocalBoundKeyOut = gOutBoundKeys + globalId * maxOutRay;

        // Prevent overwrite and better error catching
        gLocalRayOut = (maxOutRay == 0) ? nullptr : gLocalRayOut;
        gLocalAuxOut = (maxOutRay == 0) ? nullptr : gLocalAuxOut;
        gLocalBoundKeyOut = (maxOutRay == 0) ? nullptr : gLocalBoundKeyOut;

        // Actual Per-Ray Work
        WFunc(// Output
              gLocalBoundKeyOut,
              gLocalRayOut,
              gLocalAuxOut,
              maxOutRay,
              // Input as registers
              ray,
              aux,
              surface,
              rayId,
              // I-O
              localState,
              renderState,
              rng,
              // Constants
              matData,
              HitKey::FetchIdPortion(hitKey));
    }
}

// Meta Kernel for divding work.
template<class GlobalState, class LocalState,
         class RayAuxiliary, class EGroup,
         BoundaryWorkFunc<GlobalState, LocalState, RayAuxiliary, EGroup> BWFunc,
         SurfaceFunc<typename EGroup::Surface,
                     typename EGroup::HitData,
                     typename EGroup::PrimitiveData> SurfFunc>
__global__ CUDA_LAUNCH_BOUNDS_1D
void KCBoundaryWork(// Output
                    HitKey* gOutBoundKeys,
                    RayGMem* gOutRays,
                    RayAuxiliary* gOutRayAux,
                    const uint32_t maxOutRay,
                    // Input
                    const RayGMem* gInRays,
                    const RayAuxiliary* gInRayAux,
                    const PrimitiveId* gPrimitiveIds,
                    const TransformId* gTransformIds,
                    const HitStructPtr gHitStructs,
                    //
                    const HitKey* gMatIds,
                    const RayId* gRayIds,
                    // I-O
                    LocalState localState,
                    GlobalState renderState,
                    RNGGMem gRNGStates,
                    // Constants
                    const uint32_t rayCount,
                    const typename EGroup::GPUType* gEndpoints,
                    const typename EGroup::PrimitiveData primData,
                    const GPUTransformI* const* gTransforms)
{
    // Fetch Types from Template Classes
    using HitData   = typename EGroup::HitData; // HitData is defined by primitive
    using Surface   = typename EGroup::Surface; // Surface is defined by endpoint group
    using GPUType   = typename EGroup::GPUType; // Endpoint GPU Class (Derived)

    // Pre-grid stride initialization
    // RNG is allocated for each SM (not for each ray)
    RandomGPU rng(gRNGStates, LINEAR_GLOBAL_ID);

    // Grid Stride Loop
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < rayCount; globalId += blockDim.x * gridDim.x)
    {
        const RayId rayId = gRayIds[globalId];
        const HitKey hitKey = gMatIds[globalId];
        // Load Input to Registers
        const RayReg ray(gInRays, rayId);
        const RayAuxiliary aux = gInRayAux[rayId];
        const PrimitiveId primitiveId = gPrimitiveIds[rayId];
        const TransformId transformId = gTransformIds[rayId];

        // Acquire the endpoint that is hit
        HitKey::Type hitIndex = HitKey::FetchIdPortion(hitKey);
        const GPUType& gEndpoint = gEndpoints[hitIndex];

        // Skip if primitiveId is invalid only if the light is
        // primitive backed.
        // This happens when NEE generates a ray with a
        // predefined workId (which did invoke this thread)
        // However the light is missed somehow
        // (planar rays, numeric unstability etc.)
        // Because of that primitive id did not get populated
        // Skip this ray
        if(gEndpoint.IsPrimitiveBackedLight() &&
           primitiveId >= INVALID_PRIMITIVE_ID)
            continue;

        // Acquire transform for surface generation &
        // Generate surface data from hit
        const GPUTransformI& transform = *gTransforms[transformId];
        const HitData hit = gHitStructs.Ref<HitData>(rayId);
        const Surface surface = SurfFunc(hit, transform, primitiveId, primData);

        // Determine Output Location
        // Make it locally indexable
        RayGMem* gLocalRayOut = gOutRays + globalId * maxOutRay;
        RayAuxiliary* gLocalAuxOut = gOutRayAux + globalId * maxOutRay;
        HitKey* gLocalBoundKeyOut = gOutBoundKeys + globalId * maxOutRay;

        // Prevent overwrite and better error catching
        gLocalRayOut = (maxOutRay == 0) ? nullptr : gLocalRayOut;
        gLocalAuxOut = (maxOutRay == 0) ? nullptr : gLocalAuxOut;
        gLocalBoundKeyOut = (maxOutRay == 0) ? nullptr : gLocalBoundKeyOut;

        // Actual Per-Ray Work
        BWFunc(// Output
               gLocalBoundKeyOut,
               gLocalRayOut,
               gLocalAuxOut,
               maxOutRay,
               // Input as registers
               ray,
               aux,
               surface,
               rayId,
               // I-O
               localState,
               renderState,
               rng,
               // Constants
               gEndpoint);
    }
}