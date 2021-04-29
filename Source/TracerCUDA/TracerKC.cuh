#pragma once

#include "RayAuxStruct.cuh"

#include "MaterialFunctions.cuh"
#include "ImageStructs.h"
#include "RayStructs.h"
#include "GPULightI.h"
#include "EstimatorFunctions.cuh"
#include "GPUMediumVacuum.cuh"
#include "GPUSurface.h"

#include "RayLib/HemiDistribution.h"

struct DirectTracerGlobal
{
    ImageGMem<Vector4> gImage;
};

struct PathTracerGlobal : public DirectTracerGlobal
{
    const GPULightI**   lightList;
    uint32_t            totalLightCount;

    const GPUMediumI**  mediumList;
    uint32_t            totalMediumCount;

    // Options
    // Options for NEE
    bool                directLightMIS;
    bool                nee;
    int                 rrStart;
};

// No Local State
struct EmptyState {};

struct AmbientOcclusionGlobal : public DirectTracerGlobal
{
    float               maxDistance;
    bool                hitPhase;
    HitKey              aoMissKey;
};

struct PathTracerLocal
{
    bool    emptyPrimitive;
    bool    specularMaterial;
};

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
                      // I-O
                      EmptyState& gLocalState,
                      DirectTracerGlobal& gRenderState,
                      RandomGPU& rng,
                      // Constants
                      const MGroup::Data& gMatData,
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

template <class MGroup>
__device__
inline void PathLightWork(// Output
                          HitKey* gOutBoundKeys,
                          RayGMem* gOutRays,
                          RayAuxPath* gOutRayAux,
                          const uint32_t maxOutRay,
                          // Input as registers
                          const RayReg& ray,
                          const RayAuxPath& aux,
                          const typename MGroup::Surface& surface,
                          // I-O
                          PathTracerLocal& gLocalState,
                          PathTracerGlobal& gRenderState,
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
        const GPUEndpointI* endPoint = gRenderState.lightList[aux.endPointIndex];
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
        ImageAccumulatePixel(img, aux.pixelIndex, Vector4f(total, 1.0f));
    }
}

template <class MGroup>
__device__
inline void PathWork(// Output
                     HitKey* gOutBoundKeys,
                     RayGMem* gOutRays,
                     RayAuxPath* gOutRayAux,
                     const uint32_t maxOutRay,
                     // Input as registers
                     const RayReg& ray,
                     const RayAuxPath& aux,
                     const typename MGroup::Surface& surface,
                     // I-O
                     PathTracerLocal& gLocalState,
                     PathTracerGlobal& gRenderState,
                     RandomGPU& rng,
                     // Constants
                     const typename MGroup::Data& gMatData,
                     const HitKey matId,
                     const PrimitiveId primId)
{
    static constexpr Vector3 ZERO_3 = Zero3;

    // Check Material Sample Strategy
    uint32_t sampleCount = maxOutRay;
    bool specularMat = gLocalState.specularMaterial;
    static constexpr int PATH_RAY_INDEX = 0;
    static constexpr int NEE_RAY_INDEX = 1;

    // Output image
    auto& img = gRenderState.gImage;

    auto InvalidRayWrite = [&gOutRays, &gOutBoundKeys, &gOutRayAux](int index)
    {
        // Generate Dummy Ray and Terminate
        RayReg rDummy = EMPTY_RAY_REGISTER;
        rDummy.Update(gOutRays, index);
        gOutBoundKeys[index] = HitKey::InvalidKey;
        gOutRayAux[index].pixelIndex = UINT32_MAX;
     };

    // If NEE ray hits to this material
    // just skip since this is not a light material
    if(aux.type == RayType::NEE_RAY)
    {
        // Write invalids for out rays
        for(uint32_t i = 0; i < sampleCount; i++)
            InvalidRayWrite(i);
        return;
    }
    // Inputs
    const RayF& r = ray.ray;
    HitKey::Type matIndex = HitKey::FetchIdPortion(matId);
    Vector3 position = r.AdvancedPos(ray.tMax);
    Vector3 wi = -(r.getDirection().Normalize());
    const GPUMediumI& m = *(gRenderState.mediumList[aux.mediumIndex]);
    // Outputs
    RayReg rayOut = {};
    RayAuxPath auxOut = aux;
    auxOut.depth++;

    // Calculate Transmittance factor of the medium
    Vector3 transFactor = m.Transmittance(ray.tMax);
    Vector3 radianceFactor = aux.radianceFactor * transFactor;

    // Sample the emission if avail
    Vector3 emission = MGroup::Emit(// Input
                                    wi,
                                    position,
                                    m,
                                    //
                                    surface,
                                    // Constants
                                    gMatData,
                                    matIndex);
    // Only accumulate if emission has energy
    if(emission != ZERO_3)
    {
        Vector3f total = emission * radianceFactor;
        ImageAccumulatePixel(img, aux.pixelIndex, Vector4f(total, 1.0f));
    }
    
    // If this material does not require to have any samples just quit
    if(sampleCount == 0) return;

    // Sample a path from material
    RayF rayPath; float pdfPath; const GPUMediumI* outM;
    Vector3 reflectance = MGroup::Sample(// Outputs
                                         rayPath, pdfPath, outM,
                                         // Inputs
                                         wi,
                                         position,
                                         m,
                                         //
                                         surface,
                                         // I-O
                                         rng,
                                         // Constants
                                         gMatData,
                                         matIndex,
                                         0);

    // Factor the radiance of the surface
    auxOut.radianceFactor = radianceFactor * reflectance;
    // Check singularities
    auxOut.radianceFactor = (pdfPath == 0.0f) ? Zero3 : (auxOut.radianceFactor / pdfPath);

    // Change current medium of the ray
    auxOut.mediumIndex = static_cast<uint16_t>(outM->GlobalIndex());

    // Check Russian Roulette
    float avgThroughput = auxOut.radianceFactor.Dot(Vector3f(0.333f));
    if(auxOut.depth <= gRenderState.rrStart ||
       gLocalState.specularMaterial ||
       !RussianRoulette(auxOut.radianceFactor, avgThroughput, rng))
    {
        // Write Ray
        RayReg rayOut;
        rayOut.ray = rayPath;
        rayOut.tMin = 0.0f;
        rayOut.tMax = INFINITY;
        rayOut.Update(gOutRays, PATH_RAY_INDEX);

        // Write Aux
        auxOut.type = (specularMat) ? RayType::SPECULAR_PATH_RAY : RayType::PATH_RAY;
        gOutRayAux[PATH_RAY_INDEX] = auxOut;
    }
    else InvalidRayWrite(PATH_RAY_INDEX);

    // Dont launch NEE if not requested
    if((!gRenderState.nee) ||
       gLocalState.specularMaterial) return;

    // NEE Ray Generation
    float pdfLight, lDistance;
    HitKey matLight;
    Vector3 lDirection;
    uint32_t lightIndex;
    reflectance = Zero3;
    if(DoNextEventEstimation(matLight,
                             lightIndex,
                             lDirection,
                             lDistance,
                             pdfLight,
                             // Input
                             position,
                             rng,
                             //
                             gRenderState.lightList,
                             gRenderState.totalLightCount))
    {
        // Evaluate mat for this direction
        reflectance = MGroup::Evaluate(// Input
                                       lDirection,
                                       wi,
                                       position,
                                       m,
                                       //
                                       surface,
                                       // Constants
                                       gMatData,
                                       matIndex);
    }

    // Do not waste a ray if material does not reflect light
    // towards light's sampled position
    if(reflectance != Vector3(0.0f))
    {
        // Generate & Write Ray
        RayF rayNEE = RayF(lDirection, position);
        rayNEE.AdvanceSelf(MathConstants::Epsilon);
        rayOut.ray = rayNEE;
        rayOut.tMin = 0.0f;
        rayOut.tMax = lDistance;
        rayOut.Update(gOutRays, NEE_RAY_INDEX);

        // Calculate Radiance Factor
        auxOut.radianceFactor = radianceFactor * reflectance;
        // Check singularities
        auxOut.radianceFactor = (pdfLight == 0.0f) ? Zero3 : (auxOut.radianceFactor / pdfLight);
        // Write auxilary data
        auxOut.endPointIndex = lightIndex;
        auxOut.type = RayType::NEE_RAY;
        gOutRayAux[NEE_RAY_INDEX] = auxOut;
        gOutBoundKeys[NEE_RAY_INDEX] = matLight;
    }
    else InvalidRayWrite(NEE_RAY_INDEX);
}

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
                       // I-O
                       EmptyState& gLocalState,
                       AmbientOcclusionGlobal& gRenderState,
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
                   // I-O
                   EmptyState& gLocalState,
                   AmbientOcclusionGlobal& gRenderState,
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