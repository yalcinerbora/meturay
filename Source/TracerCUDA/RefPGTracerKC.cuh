#pragma once

#include "RayAuxStruct.cuh"

#include "GPULightI.h"
#include "GPUMediumI.h"
#include "GPUDirectLightSamplerI.h"

#include "RayLib/ColorConversion.h"

struct RPGTracerGlobalState
{
    // Output Image
    ImageGMem<float>                gImage;
    // Light Related
    const GPULightI**               gLightList;
    uint32_t                        totalLightCount;
    const GPUDirectLightSamplerI*   gLightSampler;
    // Medium Related
    const GPUMediumI**              mediumList;
    uint32_t                        totalMediumCount;
    // Options
    // Options for NEE
    bool                            directLightMIS;
    bool                            nee;
    int                             rrStart;
};

struct RPGTracerLocalState 
{
    bool    emptyPrimitive;
};

template <class MGroup>
__device__ __forceinline__
void RPGTracerBoundaryWork(// Output
                            HitKey* gOutBoundKeys,
                            RayGMem* gOutRays,
                            RayAuxPath* gOutRayAux,
                            const uint32_t maxOutRay,
                            // Input as registers
                            const RayReg& ray,
                            const RayAuxPath& aux,
                            const typename MGroup::Surface& surface,
                            const RayId rayId,
                            // I-O
                            RPGTracerLocalState& gLocalState,
                            RPGTracerGlobalState& gRenderState,
                            RandomGPU& rng,
                            // Constants
                            const typename MGroup::Data& gMatData,
                            const HitKey matId,
                            const PrimitiveId primId)
{
    // Check Material Sample Strategy
    assert(maxOutRay == 0);
    auto& img = gRenderState.gImage;

    // If NEE ray hits to this material
    // sample it or just sample it anyway if NEE is not activated
    bool neeMatch = (!gRenderState.nee);
    if(gRenderState.nee && aux.type == RayType::NEE_RAY)
    {
        const GPUEndpointI* endPoint = gRenderState.gLightList[aux.endPointIndex];
        PrimitiveId neePrimId = endPoint->PrimitiveIndex();
        HitKey neeKey = endPoint->BoundaryMaterial();

        // Check if NEE ray actual hit the requested light
        neeMatch = (matId.value == neeKey.value);
        if(!gLocalState.emptyPrimitive)
            neeMatch &= (primId == neePrimId);
    }
    if(neeMatch ||
       aux.type == RayType::CAMERA_RAY ||
       aux.type == RayType::SPECULAR_PATH_RAY)
    {
        const RayF& r = ray.ray;
        HitKey::Type matIndex = HitKey::FetchIdPortion(matId);
        Vector3 position = r.AdvancedPos(ray.tMax);
        const GPUMediumI& m = *(gRenderState.mediumList[aux.mediumIndex]);

        // Calculate Transmittance factor of the medium
        Vector3 transFactor = m.Transmittance(ray.tMax);
        Vector3 radianceFactor = aux.radianceFactor * transFactor;

        Vector3 emission = MGroup::Emit(// Input
                                        -r.getDirection(),
                                        position,
                                        m,
                                        //
                                        surface,
                                        // Constants
                                        gMatData,
                                        matIndex);

        // And accumulate pixel
        // and add as a sample
        Vector3f total = emission * radianceFactor;
        float luminance = Utility::RGBToLuminance(total);

        ImageAccumulatePixel(img, aux.pixelIndex, luminance);
    }
}

template <class MGroup>
__device__ __forceinline__
void RPGTracerPathWork(// Output
                       HitKey* gOutBoundKeys,
                       RayGMem* gOutRays,
                       RayAuxPath* gOutRayAux,
                       const uint32_t maxOutRay,
                       // Input as registers
                       const RayReg& ray,
                       const RayAuxPath& aux,
                       const typename MGroup::Surface& surface,
                       const RayId rayId,
                       // I-O
                       RPGTracerLocalState& gLocalState,
                       RPGTracerGlobalState& gRenderState,
                       RandomGPU& rng,
                       // Constants
                       const typename MGroup::Data& gMatData,
                       const HitKey matId,
                       const PrimitiveId primId)
{}