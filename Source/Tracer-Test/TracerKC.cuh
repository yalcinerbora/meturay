#pragma once

#include "RayAuxStruct.h"

#include "TracerLib/MaterialFunctions.cuh"
#include "TracerLib/ImageStructs.h"
#include "TracerLib/RayStructs.h"
#include "TracerLib/TextureStructs.h"
#include "TracerLib/GPULight.cuh"

struct DirectTracerGlobal
{
    ImageGMem<Vector4> gImage;
};

struct PathTracerGlobal : public DirectTracerGlobal
{
    const GPULightI**   lightList;
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
                      DirectTracerGlobal& gRenderState,
                      RandomGPU& rng,
                      // Constants
                      const MGroup::Data& gMatData,
                      const HitKey::Type& matId)
{
    // Just evaluate kernel
    //Vector3 value = MGroup::Sample();
    // Write to image
    auto& img = gRenderState.gImage;
    const RayF& r = ray.ray;
    float distance = ray.tMax - ray.tMin;

    RayF outRay; float pdf;
    Vector3 radiance = MGroup::Sample(// Outputs
                                      outRay, pdf,
                                      // Inputs
                                      -r.getDirection(),
                                      r.AdvancedPos(distance),
                                      surface,
                                      nullptr,
                                      // I-O
                                      rng,
                                      // Constants
                                      gMatData,
                                      matId,
                                      0);

    // And accumulate pixel
    ImageAccumulatePixel(img, aux.pixelId, Vector4(radiance, 1.0f));
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