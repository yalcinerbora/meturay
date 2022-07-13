#pragma once

#include "RayAuxStruct.cuh"
#include "RayStructs.h"
#include "ImageStructs.h"
#include "RNGenerator.h"

#include "ImageFunctions.cuh"

#include "RayLib/HitStructs.h"
#include "RayLib/HemiDistribution.h"

struct AmbientOcclusionGlobalState
{
    // Samples
    CamSampleGMem<Vector4f> gSamples;

    float                   maxDistance;
    bool                    hitPhase;
    HitKey                  aoMissKey;
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
                       AmbientOcclusionLocalState& localState,
                       AmbientOcclusionGlobalState& renderState,
                       RNGeneratorGPUI& rng,
                       // Constants
                       const typename MGroup::Data& gMatData,
                       const HitKey::Type matIndex)
{
    // We did not hit anything just accumulate
    Vector4f result = Vector4f(aux.aoFactor, 1.0f);
    AccumulateRaySample(renderState.gSamples,
                        aux.sampleIndex, result);
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
                   AmbientOcclusionLocalState& localState,
                   AmbientOcclusionGlobalState& renderState,
                   RNGeneratorGPUI& rng,
                   // Constants
                   const typename MGroup::Data& gMatData,
                   const HitKey::Type matIndex)
{
    // Check pass
    if(renderState.hitPhase)
    {
        // On hit phase
        // If AO kernel is called that means we hit something
        // Just skip
        // Generate Dummy Ray and Terminate
        RayReg rDummy = EMPTY_RAY_REGISTER;
        rDummy.Update(gOutRays, 0);
        gOutBoundKeys[0] = HitKey::InvalidKey;
        gOutRayAux[0].sampleIndex = UINT32_MAX;
    }
    else
    {
        // Trace phase
        // Just sample a hemispherical ray regardless of the Mat type
        // and write that ray. Also write missAO Material HitKey
        RayReg rayOut = {};
        RayAuxAO auxOut = aux;
        Vector3 position = surface.WorldPosition();

        float pdf;
        // We are at generation phase generate a ray
        Vector3 normal = surface.WorldNormal();
        Vector2 xi(rng.Uniform(), rng.Uniform());
        Vector3 direction = HemiDistribution::HemiCosineCDF(xi, pdf);
        QuatF q = Quat::RotationBetweenZAxis(normal);
        direction = q.ApplyRotation(direction);

        // Cos Theta
        float nDotL = max(normal.Dot(direction), 0.0f);

        // Ray out
        RayF ray = RayF(direction, position);
        ray.NudgeSelf(surface.WorldGeoNormal(), surface.curvatureOffset);
        // AO Calculation
        Vector3 aoMultiplier = Vector3(nDotL * MathConstants::InvPi);
        auxOut.aoFactor = aoMultiplier;
        auxOut.aoFactor = (pdf == 0.0f) ? Zero3 : (auxOut.aoFactor / pdf);

        // Finally Write
        rayOut.ray = ray;
        rayOut.tMin = 0.0f;
        rayOut.tMax = renderState.maxDistance;
        rayOut.Update(gOutRays, 0);
        // Write Aux
        gOutRayAux[0] = auxOut;
        // Here Write the Hit shaders specific boundary material
        gOutBoundKeys[0] = renderState.aoMissKey;
    }
}