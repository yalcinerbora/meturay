#pragma once

#include "RayStructs.h"
#include "ImageStructs.h"

#include "GPUMediumVacuum.cuh"

#include "RayLib/HitStructs.h"
#include "RayAuxStruct.cuh"

struct DirectTracerGlobalState
{
    ImageGMem<Vector4> gImage;
};

// No Local State
struct DirectTracerLocalState {};

template <class MGroup>
__device__
inline void DirectWork(// Output
                       HitKey* gOutBoundKeys,
                       RayGMem* gOutRays,
                       RayAuxBasic* gOutRayAux,
                       const uint32_t maxOutRay,
                       // Input as registers
                       const RayReg& ray,
                       const RayAuxBasic& aux,
                       const typename MGroup::Surface& surface,
                       // I-O
                       DirectTracerLocalState& gLocalState,
                       DirectTracerGlobalState& gRenderState,
                       RandomGPU& rng,
                       // Constants
                       const typename MGroup::Data& gMatData,
                       const HitKey matId,
                       const PrimitiveId primId)
{
    // Just evaluate kernel
    //Vector3 value = MGroup::Sample();
    // Write to image
    auto& img = gRenderState.gImage;
    const RayF& r = ray.ray;
    HitKey::Type matIndex = HitKey::FetchIdPortion(matId);

    const GPUMediumVacuum m(0);
    const GPUMediumI* outM;
    RayF outRay; float pdf;

    Vector3 emission = MGroup::Emit(-r.getDirection(),
                                    r.AdvancedPos(ray.tMax),
                                    m,
                                    //
                                    surface,
                                    //
                                    gMatData,
                                    matIndex);

    Vector3 radiance = MGroup::Sample(// Outputs
                                      outRay, pdf, outM,
                                      // Inputs
                                      -r.getDirection(),
                                      r.AdvancedPos(ray.tMax),
                                      m,
                                      //
                                      surface,
                                      // I-O
                                      rng,
                                      // Constants
                                      gMatData,
                                      matIndex,
                                      0);

    radiance = (pdf == 0.0f) ? Zero3 : (radiance / pdf);
    radiance += emission;

    //printf("r(%f, %f, %f), e(%f, %f, %f), pdf %f\n",
    //       radiance[0], radiance[1], radiance[2],
    //       emission[0], emission[1], emission[2],
    //       pdf);

    // And accumulate pixel
    ImageAccumulatePixel(img, aux.pixelIndex, Vector4(radiance, 1.0f));
}
