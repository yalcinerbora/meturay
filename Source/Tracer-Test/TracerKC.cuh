#pragma once

#include "RayAuxStruct.h"

#include "TracerLib/MaterialFunctions.cuh"
#include "TracerLib/ImageStructs.h"
#include "TracerLib/RayStructs.h"
#include "TracerLib/TextureStructs.h"
#include "TracerLib/GPULight.cuh"

struct BasicTracerGlobal
{
    ImageGMem<Vector4> gImage;
};

struct PathTracerGlobal : public BasicTracerGlobal
{
    const GPULightI*    lightList;
    uint32_t            totalLightCount;
};

// No Local State
struct EmptyState {};

template <class MGroup>
__device__
inline void BasicWork(// Output
                      HitKey* gOutBoundKeys,
                      RayGMem* gOutRays,
                      RayAuxBasic* gOutRayAux,
                      const uint32_t maxOutRay,
                      // Input as registers                         
                      const RayReg& ray,
                      const RayAuxBasic& aux,
                      const MGroup::Surface& surface,
                      const UVList* uvs,
                      // I-O
                      EmptyState& gLocalState,
                      BasicTracerGlobal& gRenderState,
                      RandomGPU& rng,
                      // Constants
                      const MGroup::Data& gMatData,
                      const HitKey::Type& matId)
{
    // Just evaluate kernel
    //Vector3 value = MGroup::Sample();
    // Write to image

}

template <class MGroup>
__device__
inline void PathWork(// Output
                     HitKey* gOutBoundKeys,
                     RayGMem* gOutRays,
                     RayAuxBasic* gOutRayAux,
                     const uint32_t maxOutRay,
                     // Input as registers                         
                     const RayReg& ray,
                     const RayAuxBasic& aux,
                     const MGroup::Surface& surface,
                     const UVList* uvs,
                     // I-O
                     EmptyState& gLocalState,
                     PathTracerGlobal& gRenderState,
                     RandomGPU& rng,
                     // Constants
                     const MGroup::Data& gMatData,
                     const HitKey::Type& matId)
{
    //MGroup::Sample()
}