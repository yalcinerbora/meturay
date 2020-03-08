#pragma once

#include "RayLib/HitStructs.h"
#include "RayStructs.h"
#include "Random.cuh"
#include "ImageStructs.h"


// Surface Functions are responsible for
// generating surface structures from the hit structures
// Surface structures hold various surface data (uv, normal most of the time;
// maybe tangent if a normal map is present over a surface)
//
// Surfaces are deemed by the materials
// (i.e. a material may not require uv if a texture is present)
//
// This function is provided by the "WorkBatch" class 
// meaning different WorkBatch Class is generated for different
// primitive/material pairs
template <class MGroup, class PGroup>
using SurfaceFunc = MGroup::Surface(*)(const PGroup::PrimitiveData&,
                                       const PGroup::HitData&,
                                       PrimitiveId);

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
                         const RayAuxiliary & aux,
                         const MGroup::Surface& surface,
                         const UVList * uvs,
                         // I-O
                         LocalState& gLocalState,
                         GlobalState& gRenderState,
                         RandomGPU& rng,
                         // Constants
                         const MGroup::Data& gMatData,
                         const HitKey::Type& matId);

// Meta Kernel for divding work.
template<class GlobalState, class LocalState,
         class RayAuxiliary, class PGroup, class MGroup,
         WorkFunc<GlobalState, LocalState, RayAuxiliary, MGroup> WFunc, 
         SurfaceFunc<MGroup, PGroup> SurfFunc>
 __global__ //__launch_bounds__(StaticThreadPerBlock1D)
void KCWork(// Output
            HitKey* gOutBoundKeys,
            RayGMem* gOutRays,
            RayAuxiliary* gOutRayAux,
            const uint32_t maxOutRay,
            // Input
            const RayGMem* gInRays,
            const RayAuxiliary* gInRayAux,
            const PrimitiveId* gPrimitiveIds,
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
            const PGroup::Data primData)
{
    // Fetch Types from Template Classes
    using HitData = typename PGroup::HitData;               // HitData is defined by primitive
    using Surface = typename MGroup::Surface;               // Surface is defined by material group    

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
        const RayAuxiliary aux = gInRayAux[rayId];
        const PrimitiveId gPrimitiveId = gPrimitiveIds[rayId];

        // Generate surface data from hit
        const HitData hit = gHitStructs.Ref<HitData>(rayId);
        const Surface surface = SurfFunc(primData, hit, gPrimitiveId);

        // Determine Output Location
        // Make it locally indexable
        RayGMem* gLocalRayOut = gOutRays + globalId * maxOutRay;
        RayAuxiliary* gLocalAuxOut = gOutRayAux + globalId * maxOutRay;
        HitKey* gLocalBoundKeyOut = gOutBoundKeys + globalId * maxOutRay;

        // Actual Per-Ray Work
        WFunc(// Output
              gLocalBoundKeyOut,
              gLocalRayOut,
              gLocalAuxOut,
              maxOutRay,
              // Input as registers
              ray,
              surface,
              aux,
              // I-O
              gLocalState,
              gRenderState,
              rng,
              // Constants
              matData,
              HitKey::FetchIdPortion(hitKey));
    }
}

//
//template <class TLogic, class ELogic, class Surface, class MaterialData>
//using ShadeFunc = void(*)(// Output
//                          ImageGMem<Vector4f> gImage,
//                          //
//                          HitKey* gOutBoundKeys,
//                          RayGMem* gOutRays,
//                          TLogic::RayAuxData* gOutRayAux,
//                          const uint32_t maxOutRay,
//                          // Input as registers
//                          const RayReg& ray,
//                          const Surface& surface,
//                          const TLogic::RayAuxData& aux,
//                          // RNG
//                          RandomGPU& rng,
//                          // Estimator
//                          const ELogic::EstimatorData& estData,
//                          // Input as global memory
//                          const MaterialData& gMatData,
//                          const HitKey::Type& matId);
//
//template <class TLogic, class ELogic, class MGroup, class PGroup,
//          SurfaceFunc<MGroup, PGroup> SurfFunc>
//__global__ __launch_bounds__(StaticThreadPerBlock1D)
//void KCMaterialShade(// Output
//                     ImageGMem<Vector4f> gImage,
//                     //
//                     HitKey* gOutBoundKeys,
//                     RayGMem* gOutRays,
//                     TLogic::RayAuxData* gOutRayAux,
//                     const uint32_t maxOutRay,
//                     // Input
//                     const RayGMem* gInRays,
//                     const TLogic::RayAuxData* gInRayAux,
//                     const PrimitiveId* gPrimitiveIds,
//                     const HitStructPtr gHitStructs,
//                     //
//                     const HitKey* gMatIds,
//                     const RayId* gRayIds,
//                     //
//                     const uint32_t rayCount,
//                     RNGGMem gRNGStates,
//                     // Estimator
//                     const ELogic::EstimatorData est,
//                     // Material Related
//                     const MGroup::MaterialData matData,
//                     // Primitive Related
//                     const PGroup::PrimitiveData primData)
//{
//    // Fetch Types from Template Classes
//    using HitData = typename PGroup::HitData;               // HitData is defined by primitive
//    using Surface = typename MGroup::Surface;               // Surface is defined by material group
//    using RayAuxData = typename TLogic::RayAuxData;         // Hit register defined by primitive
//
//    // Pre-grid stride loop
//    // RNG is allocated for each SM (not for each thread)
//    RandomGPU rng(gRNGStates.state);
//
//    // Grid Stride Loop
//    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
//        globalId < rayCount; globalId += blockDim.x * gridDim.x)
//    {
//        const RayId rayId = gRayIds[globalId];
//        const HitKey hitKey = gMatIds[globalId];
//
//        // Load Input to Registers
//        const RayReg ray(gInRays, rayId);
//        const RayAuxData aux = gInRayAux[rayId];
//        const PrimitiveId gPrimitiveId = gPrimitiveIds[rayId];
//
//        // Generate surface data from hit
//        const HitData hit = gHitStructs.Ref<HitData>(rayId);
//        const Surface surface = SurfFunc(primData, hit,
//                                         gPrimitiveId);
//
//        // Determine Output Location
//        RayGMem* gLocalRayOut = gOutRays + globalId * maxOutRay;
//        RayAuxData* gLocalAuxOut = gOutRayAux + globalId * maxOutRay;
//        HitKey* gLocalBoundKeyOut = gOutBoundKeys + globalId * maxOutRay;
//        // Actual Shading
//        MGroup::ShadeFunc(// Output
//                          gImage,
//                          //
//                          gLocalBoundKeyOut,
//                          gLocalRayOut,
//                          gLocalAuxOut,
//                          maxOutRay,
//                          // Input as registers
//                          ray,
//                          surface,
//                          aux,
//                          // RNG
//                          rng,
//                          // Estimator
//                          est,
//                          // Input as global memory
//                          matData,
//                          HitKey::FetchIdPortion(hitKey));
//    }
//}