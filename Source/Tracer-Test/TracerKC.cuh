#pragma once

#include "RayAuxStruct.h"

#include "TracerLib/MaterialFunctions.cuh"
#include "TracerLib/ImageStructs.h"
#include "TracerLib/RayStructs.h"
#include "TracerLib/TextureStructs.h"
#include "TracerLib/GPULight.cuh"
#include "TracerLib/EstimatorFunctions.cuh"

struct DirectTracerGlobal
{
    ImageGMem<Vector4> gImage;
};

struct PathTracerGlobal : public DirectTracerGlobal
{
    const GPULightI**   lightList;
    uint32_t            totalLightCount;

    const GPUMedium*    mediumList;
    uint32_t            totalMediumCount;

    // Options
    // Options for NEE
    bool                nee;
    int                 rrStart;
    float               rrFactor;
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
                      const TexCoords* uvs,
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
    float distance = ray.tMax - ray.tMin;
    HitKey::Type matIndex = HitKey::FetchIdPortion(matId);

    GPUMedium m, outM;
    RayF outRay; float pdf;
    Vector3 radiance = MGroup::Sample(// Outputs
                                      outRay, pdf, outM,
                                      // Inputs
                                      -r.getDirection(),
                                      r.AdvancedPos(distance),
                                      m,
                                      //
                                      surface,
                                      nullptr,
                                      // I-O
                                      rng,
                                      // Constants
                                      gMatData,
                                      matIndex,
                                      0);

    // And accumulate pixel
    ImageAccumulatePixel(img, aux.pixelIndex, Vector4(radiance, 1.0f));
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
                     const TexCoords* uvs,
                     // I-O
                     EmptyState& gLocalState,
                     PathTracerGlobal& gRenderState,
                     RandomGPU& rng,
                     // Constants
                     const typename MGroup::Data& gMatData,
                     const HitKey matId,
                     const PrimitiveId primId)
{
    assert(maxOutRay == (gRenderState.nee) ? 2 : 1);

    static constexpr int NEE_RAY_INDEX = 1;
    static constexpr int PATH_RAY_INDEX = 0;

    // Inputs
    auto& img = gRenderState.gImage;
    const RayF& r = ray.ray;
    float distance = ray.tMax - ray.tMin;
    HitKey::Type matIndex = HitKey::FetchIdPortion(matId);
    Vector3 position = r.AdvancedPos(distance);
    GPUMedium m = gRenderState.mediumList[aux.mediumIndex];
    // Outputs
    RayReg rayOut = {};
    RayAuxPath auxOutPath = auxIn;
    RayAuxPath auxOutNEE = auxIn;

    auxOutPath.depth++;
    auxOutNEE.depth++;
    auxOutPath.type = RayType::PATH_RAY;
    auxOutNEE.type = RayType::NEE_RAY;
    auxOutNEE.type = RayType::NEE_RAY;

    PrimitiveId neePrimId = gRenderState.lightList[aux.endPointIndex]->BoundaryMaterial();
    HitKey neeKey = gRenderState.lightList[aux.endPointIndex]->Primitive();

    // Apply Decay of the medium
    Vector3 decay = m.Transmittance(distance);
    auxOutPath.radianceFactor *= decay;
    auxOutNEE.radianceFactor *= decay;

    // End Case Check (We finally hit a light)
    bool neeLight = (aux.type == RayType::NEE_RAY &&
                     primId == neePrimId &&
                     matId == neeKey);
    bool wrongNEELight = (aux.type == RayType::NEE_RAY &&
                          primId != neePrimId ||
                          matId != neeKey);
    bool nonNEELight = (!gRenderState.nee &&
                        MGroup::IsBoundaryMat());

    if(neeLight || nonNEELight)
    {
        // We found the light that we required to sample
        // Evaluate
        Vector3 emission = MGroup::EmitFunc(// Input
                                            -r.getDirection(),
                                            position,
                                            m,
                                            //
                                            surface,
                                            nullptr,
                                            // Constants
                                            gMatData,
                                            matIndex);
        // And accumulate pixel
        Vector3f total = emission * aux.radianceFactor;
        ImageAccumulatePixel(img, aux.pixelIndex, Vector4f(total, 1.0f));
    }
    if(wrongNEELight || neeLight || nonNEELight)
    {        
        // Generate Dummy Ray and Terminate
        RayReg rDummy = EMPTY_RAY_REGISTER;
        rDummy.Update(gOutRays, PATH_RAY_INDEX);
        rDummy.Update(gOutRays, NEE_RAY_INDEX);
        gOutBoundKeys[PATH_RAY_INDEX] = HitKey::InvalidKey;
        gOutBoundKeys[NEE_RAY_INDEX] = HitKey::InvalidKey;
        return;
    }

    // Path Ray
    // Sample a path
    RayF rayPath; float pdfPath; GPUMedium outM;
    Vector3 reflectance = MGroup::Sample(// Outputs
                                         rayPath, pdfPath, outM,
                                         // Inputs
                                         -r.getDirection(),
                                         position,
                                         m,
                                         //
                                         surface,
                                         nullptr,
                                         // I-O
                                         rng,
                                         // Constants
                                         gMatData,
                                         matIndex,
                                         0);

    // Factor the radiance of the surface
    auxOutPath.radianceFactor *= (reflectance / pdfPath);
    // Change current medium of the ray
    auxOutPath.mediumIndex = static_cast<uint16_t>(outM.ID());

    // Check Russian Roulette
    float avgThroughput = auxOutPath.radianceFactor.Dot(Vector3f(gRenderState.rrFactor));
    if(auxIn.depth <= gRenderState.rrStart &&
       !RussianRoulette(auxOutPath.radianceFactor, avgThroughput, rng))
    {
        // Write Ray        
        rayPath.Advance(MathConstants::Epsilon);
        rayOut.ray = rayPath;
        rayOut.tMin = 0.0f;
        rayOut.tMax = INFINITY;

        // Write to GMem
        rayOut.Update(gOutRays, PATH_RAY_INDEX);
        gOutRayAux[PATH_RAY_INDEX] = auxOutPath;
    }
    else
    {
        // Generate Dummy Ray and Terminate
        RayReg rDummy = EMPTY_RAY_REGISTER;
        rDummy.Update(gOutRays, PATH_RAY_INDEX);
        gOutBoundKeys[PATH_RAY_INDEX] = HitKey::InvalidKey;
    }
    // NEE Ray
    // Launch a NEE Ray if requested
    float pdfLight, lDistance;
    HitKey matLight;
    Vector3 lDirection;
    uint32_t lightIndex;
    if(gRenderState.nee &&
       NextEventEstimation(matLight,
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
        // Advance slightly to prevent self intersection
        RayF rayNEE = RayF(lDirection, position);
        rayNEE.Advance(MathConstants::Epsilon); 
        rayOut.ray = rayNEE;
        rayOut.tMin = 0.0f;
        rayOut.tMax = lDistance - MathConstants::Epsilon;

        // Evaluate mat for this direction
        Vector3 reflectance = MGroup::Evaluate(// Input
                                               lDirection,
                                               -r.getDirection(),
                                               m,
                                               //
                                               position,
                                               surface,
                                               nullptr,
                                               // Constants
                                               gMatData,
                                               matIndex);

        // Incorporate for Radiance Factor
        auxOutNEE.radianceFactor *= reflectance / pdfLight;
        // Set Endpoint Index
        auxOutNEE.endPointIndex = lightIndex;

        // Write to global memory
        rayOut.Update(gOutRays, NEE_RAY_INDEX);
        gOutRayAux[NEE_RAY_INDEX] = auxOutNEE;
        gOutBoundKeys[NEE_RAY_INDEX] = matLight;
    }
    else
    {
        // Generate Dummy Ray and Terminate
        RayReg rDummy = EMPTY_RAY_REGISTER;
        rDummy.Update(gOutRays, NEE_RAY_INDEX);
        gOutBoundKeys[NEE_RAY_INDEX] = HitKey::InvalidKey;
    }    
}