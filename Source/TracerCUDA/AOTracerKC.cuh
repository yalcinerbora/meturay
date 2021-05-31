#pragma once

#include "RayAuxStruct.cuh"
#include "RayStructs.h"
#include "ImageStructs.h"
#include "Random.cuh"

#include "ImageFunctions.cuh"

#include "RayLib/HitStructs.h"
#include "RayLib/HemiDistribution.h"

struct AmbientOcclusionGlobalState
{
    ImageGMem<Vector4>  gImage;

    float               maxDistance;
    bool                hitPhase;
    HitKey              aoMissKey;
};

struct AmbientOcclusionLocalState {};

template <class MGroup>
__device__
inline void AOMissWork(// Output
                       HitKey* gOutBoundKeys,
                       RayGMem* gOutRays,
                       RayAuxAO* gOutRayAux,
                       const uint32_t maxOutRay,
                       // Input as registers
                       const RayReg& ray,
                       const RayAuxAO& aux,
                       const typename MGroup::Surface& surface,
                       const RayId rayId,
                       // I-O
                       AmbientOcclusionLocalState& gLocalState,
                       AmbientOcclusionGlobalState& gRenderState,
                       RandomGPU& rng,
                       // Constants
                       const typename MGroup::Data& gMatData,
                       const HitKey matId,
                       const PrimitiveId primId)
{
    // We did not hit anything just accumulate accumulate
    auto& img = gRenderState.gImage;
    Vector4f result = Vector4f(aux.aoFactor, 1.0f);
    ImageAccumulatePixel(img, aux.pixelIndex, result);
}

template <class MGroup>
__device__
inline void AOWork(// Output
                   HitKey* gOutBoundKeys,
                   RayGMem* gOutRays,
                   RayAuxAO* gOutRayAux,
                   const uint32_t maxOutRay,
                   // Input as registers
                   const RayReg& ray,
                   const RayAuxAO& aux,
                   const typename MGroup::Surface& surface,
                   const RayId rayId,
                   // I-O
                   AmbientOcclusionLocalState& gLocalState,
                   AmbientOcclusionGlobalState& gRenderState,
                   RandomGPU& rng,
                   // Constants
                   const typename MGroup::Data& gMatData,
                   const HitKey matId,
                   const PrimitiveId primId)
{
    // Check pass
    if(gRenderState.hitPhase)
    {
        // On hit phase
        // If AO kernel is called that means we hit something
        // Just skip
        // Generate Dummy Ray and Terminate
        RayReg rDummy = EMPTY_RAY_REGISTER;
        rDummy.Update(gOutRays, 0);
        gOutBoundKeys[0] = HitKey::InvalidKey;
        gOutRayAux[0].pixelIndex = UINT32_MAX;
    }
    else
    {
        // Trace phase
        // Just sample a hemispherical ray regardless of the Mat type
        // and write that ray. Also write missAO Material HitKey
        RayReg rayOut = {};
        RayAuxAO auxOut = aux;
        Vector3 position = ray.ray.AdvancedPos(ray.tMax);

        float pdf;
        // We are at generation phase generate a ray
        Vector3 normal = GPUSurface::NormalWorld(surface.worldToTangent);
        Vector2 xi(GPUDistribution::Uniform<float>(rng),
                   GPUDistribution::Uniform<float>(rng));
        Vector3 direction = HemiDistribution::HemiCosineCDF(xi, pdf);
        QuatF q = Quat::RotationBetweenZAxis(normal);
        direction = q.ApplyRotation(direction);

        // Cos Tetha
        float nDotL = max(normal.Dot(direction), 0.0f);

        // Ray out
        Vector3 outPos = position + normal * MathConstants::Epsilon;
        RayF ray = RayF(direction, outPos);
        // AO Calculation
        Vector3 aoMultiplier = Vector3(nDotL * MathConstants::InvPi);
        auxOut.aoFactor = aoMultiplier;
        auxOut.aoFactor = (pdf == 0.0f) ? Zero3 : (auxOut.aoFactor / pdf);

        // Finally Write
        rayOut.ray = ray;
        rayOut.tMin = 0.0f;
        rayOut.tMax = gRenderState.maxDistance;
        rayOut.Update(gOutRays, 0);
        // Write Aux
        gOutRayAux[0] = auxOut;
        // Here Write the Hit shaders specific boundary material
        gOutBoundKeys[0] = gRenderState.aoMissKey;
    }
}