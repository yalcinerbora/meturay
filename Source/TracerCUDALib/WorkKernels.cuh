#pragma once

#include "RayLib/HitStructs.h"
#include "RayStructs.h"
#include "Random.cuh"
#include "ImageStructs.h"
#include "TextureStructs.h"
#include "GPUPrimitiveP.cuh"

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
                         const MGroup::Surface& surface,
                         const TexCoords* uvs,
                         // I-O
                         LocalState& gLocalState,
                         GlobalState& gRenderState,
                         RandomGPU& rng,
                         // Constants
                         const MGroup::Data& gMatData,
                         const HitKey matId,
                         const PrimitiveId primId);

// Meta Kernel for divding work.
template<class GlobalState, class LocalState,
         class RayAuxiliary, class PGroup, class MGroup,
         WorkFunc<GlobalState, LocalState, RayAuxiliary, MGroup> WFunc, 
         SurfaceFunc<MGroup::Surface, 
                     PGroup::HitData, 
                     PGroup::PrimitiveData> SurfFunc>
__global__ __launch_bounds__(StaticThreadPerBlock1D)
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
            LocalState gLocalState,
            GlobalState gRenderState,
            RNGGMem gRNGStates,
            // Constants
            const uint32_t rayCount,
            const MGroup::Data matData,
            const PGroup::PrimitiveData primData,
            const GPUTransformI* const* gTransforms)
{
    // Fetch Types from Template Classes
    using HitData = typename PGroup::HitData;   // HitData is defined by primitive
    using Surface = typename MGroup::Surface;   // Surface is defined by material group    

    // Pre-grid stride loop
    // RNG is allocated for each SM (not for each thread)
    RandomGPU rng(gRNGStates.state, LINEAR_GLOBAL_ID);
   
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
              nullptr,
              // I-O
              gLocalState,
              gRenderState,
              rng,
              // Constants
              matData,
              hitKey,
              primitiveId);
    }
}