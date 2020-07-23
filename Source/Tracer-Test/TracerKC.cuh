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

struct PathTracerLocal
{
    uint32_t materialSampleCount;
    bool emissiveMaterial;
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
    HitKey::Type matIndex = HitKey::FetchIdPortion(matId);

    GPUMedium m, outM;
    RayF outRay; float pdf;
    Vector3 radiance = MGroup::Sample(// Outputs
                                      outRay, pdf, outM,
                                      // Inputs
                                      -r.getDirection(),
                                      r.AdvancedPos(ray.tMax),
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
inline void PathLightWork(// Output
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
                          PathTracerLocal& gLocalState,
                          PathTracerGlobal& gRenderState,
                          RandomGPU& rng,
                          // Constants
                          const typename MGroup::Data& gMatData,
                          const HitKey matId,
                          const PrimitiveId primId)
{
    // Check Material Sample Strategy
    assert(gLocalState.materialSampleCount == 0);
    auto& img = gRenderState.gImage;

    // If NEE ray hit to these material
    // sample it
    // or just sample it if NEE is not activated
    bool neeMatch = (!gRenderState.nee);
    if(gRenderState.nee && aux.type == RayType::NEE_RAY)
    {
        //
        const GPUEndpointI* endPoint = gRenderState.lightList[aux.endPointIndex];
        PrimitiveId neePrimId = endPoint->Primitive();
        HitKey neeKey = endPoint->BoundaryMaterial();

        // Check if NEE ray actual hit the sampled light
        neeMatch = (primId == neePrimId && matId.value == neeKey.value);
    }
    if(neeMatch)
    {
        const RayF& r = ray.ray;
        HitKey::Type matIndex = HitKey::FetchIdPortion(matId);
        Vector3 position = r.AdvancedPos(ray.tMax);
        GPUMedium m = gRenderState.mediumList[aux.mediumIndex];

        Vector3 emission = MGroup::Emit(// Input
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
        // and add as a sample
        Vector3f total = emission * aux.radianceFactor;
        ImageAccumulatePixel(img, aux.pixelIndex, Vector4f(total, 1.0f));
        ImageAddSample(gRenderState.gImage, aux.pixelIndex, 1);
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
                     const TexCoords* uvs,
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
    uint32_t sampleCount = gLocalState.materialSampleCount;
    bool emissiveMaterial = gLocalState.emissiveMaterial;
    static int PATH_RAY_INDEX = 0;
    static int NEE_RAY_INDEX = 1;
    
    // Output image
    auto& img = gRenderState.gImage;

    auto InvalidRayWrite = [&gOutRays, &gOutBoundKeys](int index)
    {
        // Generate Dummy Ray and Terminate
        RayReg rDummy = EMPTY_RAY_REGISTER;
        rDummy.Update(gOutRays, index);
        gOutBoundKeys[index] = HitKey::InvalidKey;
    };

    // If NEE ray hit to these material
    // which can not be sampled with NEE ray just skip
    if(aux.type == RayType::NEE_RAY)
    {
        // Write invalids for out rays
        for(uint32_t i = 0; i < sampleCount; i++)
            InvalidRayWrite(i);

        // We still did a sample (although it returns nothing)
        // Increment sample count for that pixel
        ImageAddSample(img, aux.pixelIndex, 1);

        // All done
        return;
    }

    // Now do work normally    
    // Inputs    
    const RayF& r = ray.ray;
    HitKey::Type matIndex = HitKey::FetchIdPortion(matId);
    Vector3 position = r.AdvancedPos(ray.tMax);
    GPUMedium m = gRenderState.mediumList[aux.mediumIndex];
    // Outputs
    RayReg rayOut = {};
    RayAuxPath auxOut = aux;
    auxOut.depth++;
    
    // Sample the Emission if avail
    if(emissiveMaterial)
    {
        Vector3 emission = MGroup::Emit(// Input
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
        // and add as a sample
        Vector3f total = emission * aux.radianceFactor;
        ImageAccumulatePixel(img, aux.pixelIndex, Vector4f(total, 1.0f));
        ImageAddSample(img, aux.pixelIndex, 1);
    }

    // If this material does not require to have any samples just quit
    if(sampleCount == 0) return;


    // TODO: Loop sample all of the sample strategies of material
    // on the porper implementation
    // Now material needs to be sampled
    // Sample a path from material
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
    auxOut.radianceFactor = aux.radianceFactor * (reflectance / pdfPath);
    // Change current medium of the ray
    auxOut.mediumIndex = static_cast<uint16_t>(outM.ID());

    // Check Russian Roulette
    float avgThroughput = auxOut.radianceFactor.Dot(Vector3f(gRenderState.rrFactor));
    if(auxOut.depth <= gRenderState.rrStart &&
       !RussianRoulette(auxOut.radianceFactor, avgThroughput, rng))
    {
        // Write Ray
        RayReg rayOut;
        rayPath.AdvanceSelf(MathConstants::Epsilon);
        rayOut.ray = rayPath;
        rayOut.tMin = 0.0f;
        rayOut.tMax = INFINITY;
        rayOut.Update(gOutRays, PATH_RAY_INDEX);
        
        // Write Aux
        auxOut.type = RayType::PATH_RAY;
        gOutRayAux[PATH_RAY_INDEX] = auxOut;
    }
    else
    {
        // Terminate
        InvalidRayWrite(PATH_RAY_INDEX);
        ImageAddSample(img, aux.pixelIndex, 1);
    }

    // Dont launch NEE if not requested
    if(!gRenderState.nee) return;

    // NEE Ray Generation
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
        //Vector3 p = position + 0.001f * lDirection;

        //printf("(%f, %f, %f), (%f, %f, %f)\n",
        //       position[0], position[1], position[2],
        //       //lDirection[0], lDirection[1], lDirection[2],
        //       p[0], p[1], p[2]);
        //       //r.getPosition()[0], r.getPosition()[1], r.getPosition()[2]);

        RayF rayNEE = RayF(lDirection, position);
        rayNEE.AdvanceSelf(MathConstants::Epsilon);
        rayOut.ray = rayNEE;
        rayOut.tMin = 0.0f;
        rayOut.tMax = lDistance + MathConstants::Epsilon;
        rayOut.Update(gOutRays, NEE_RAY_INDEX);

        // Evaluate mat for this direction
        Vector3 reflectance = MGroup::Evaluate(// Input
                                               lDirection,
                                               -r.getDirection(),
                                               position,
                                               m,
                                               //
                                               surface,
                                               nullptr,
                                               // Constants
                                               gMatData,
                                               matIndex);

        // Gen aux out
        auxOut.radianceFactor = aux.radianceFactor * reflectance / pdfLight;
        auxOut.endPointIndex = lightIndex;
        auxOut.type = RayType::NEE_RAY;
        gOutRayAux[NEE_RAY_INDEX] = auxOut;
        gOutBoundKeys[NEE_RAY_INDEX] = matLight;
    }
    else InvalidRayWrite(NEE_RAY_INDEX);
    
    ////PrimitiveId neePrimId = gRenderState.lightList[aux.endPointIndex]->Primitive();
    ////HitKey neeKey = gRenderState.lightList[aux.endPointIndex]->BoundaryMaterial(); 

    ////// Apply Decay of the medium
    //////Vector3 decay = m.Transmittance((ray.tMax - ray.tMin));
    //////auxOutPath.radianceFactor *= decay;
    //////auxOutNEE.radianceFactor *= decay;

    ////// End Case Check (We finally hit a light)
    
    ////bool neeLight = (aux.type == RayType::NEE_RAY && neeMatch);
    ////bool wrongNEELight = (aux.type == RayType::NEE_RAY && !neeMatch);
    ////bool nonNEELight = (!gRenderState.nee &&
    ////                    MGroup::IsEmissive(gMatData, matIndex));

    ////if(aux.type == RayType::NEE_RAY)
    ////    printf("(%d), %d, %d, %d\n",
    ////           static_cast<uint32_t>(aux.type),
    ////           neeLight, wrongNEELight, nonNEELight);

    ////if(aux.type == RayType::NEE_RAY)
    ////    printf("(%x, %x), (%llu, %llu)\n",
    ////           neeKey.value, matId.value,
    ////           neePrimId, primId);

    ////printf("%p\n", gRenderState.lightList);
    ////printf("%p\n", gRenderState.lightList[0]);
    ////if(aux.type == RayType::NEE_RAY &&
    ////   HitKey::FetchBatchPortion(matId) == 1)
    ////{
    ////    __brkpt();
    ////}

    //if(neeLight || nonNEELight)
    //{
    //    // We found the light that we required to sample
    //    // Evaluate
    //    Vector3 emission = MGroup::Emit(// Input
    //                                    -r.getDirection(),
    //                                    position,
    //                                    m,
    //                                    //
    //                                    surface,
    //                                    nullptr,
    //                                    // Constants
    //                                    gMatData,
    //                                    matIndex);
    //    // And accumulate pixel
    //    Vector3f total = emission* aux.radianceFactor;
    //    ImageAccumulatePixel(img, aux.pixelIndex, Vector4f(total, 1.0f));
    //}
    //if(wrongNEELight || neeLight || nonNEELight)
    //{  

    //    // Generate Dummy Ray and Terminate
    //    RayReg rDummy = EMPTY_RAY_REGISTER;
    //    rDummy.Update(gOutRays, PATH_RAY_INDEX);
    //    rDummy.Update(gOutRays, NEE_RAY_INDEX);
    //    gOutBoundKeys[PATH_RAY_INDEX] = HitKey::InvalidKey;
    //    gOutBoundKeys[NEE_RAY_INDEX] = HitKey::InvalidKey;
    //    return;
    //}

    //// Path Ray
    //// Sample a path
    //RayF rayPath; float pdfPath; GPUMedium outM;
    //Vector3 reflectance = MGroup::Sample(// Outputs
    //                                     rayPath, pdfPath, outM,
    //                                     // Inputs
    //                                     -r.getDirection(),
    //                                     position,
    //                                     m,
    //                                     //
    //                                     surface,
    //                                     nullptr,
    //                                     // I-O
    //                                     rng,
    //                                     // Constants
    //                                     gMatData,
    //                                     matIndex,
    //                                     0);

    //// Factor the radiance of the surface
    //auxOutPath.radianceFactor *= (reflectance / pdfPath);
    //// Change current medium of the ray
    //auxOutPath.mediumIndex = static_cast<uint16_t>(outM.ID());

    ////// Check Russian Roulette
    ////float avgThroughput = auxOutPath.radianceFactor.Dot(Vector3f(gRenderState.rrFactor));
    ////if(auxOutPath.depth <= gRenderState.rrStart &&
    ////   !RussianRoulette(auxOutPath.radianceFactor, avgThroughput, rng))
    ////{
    ////    // Write Ray        
    ////    rayPath.AdvanceSelf(MathConstants::Epsilon);
    ////    rayOut.ray = rayPath;
    ////    rayOut.tMin = 0.0f;
    ////    rayOut.tMax = INFINITY;

    ////    // Write to GMem
    ////    rayOut.Update(gOutRays, PATH_RAY_INDEX);
    ////    gOutRayAux[PATH_RAY_INDEX] = auxOutPath;
    ////}
    ////else
    //{
    //    // Generate Dummy Ray and Terminate
    //    RayReg rDummy = EMPTY_RAY_REGISTER;
    //    rDummy.Update(gOutRays, PATH_RAY_INDEX);
    //    gOutBoundKeys[PATH_RAY_INDEX] = HitKey::InvalidKey;
    //}
    //// NEE Ray
    //// Launch a NEE Ray if requested
    //float pdfLight, lDistance;
    //HitKey matLight;
    //Vector3 lDirection;
    //uint32_t lightIndex;
    //if(gRenderState.nee &&
    //   NextEventEstimation(matLight,
    //                       lightIndex,
    //                       lDirection,
    //                       lDistance,
    //                       pdfLight,
    //                       // Input
    //                       position,
    //                       rng,
    //                       //
    //                       gRenderState.lightList,
    //                       gRenderState.totalLightCount))
    //{   
    //    // Advance slightly to prevent self intersection
    //    //Vector3 p = position + 0.001f * lDirection;

    //    //printf("(%f, %f, %f), (%f, %f, %f)\n",
    //    //       position[0], position[1], position[2],
    //    //       //lDirection[0], lDirection[1], lDirection[2],
    //    //       p[0], p[1], p[2]);
    //    //       //r.getPosition()[0], r.getPosition()[1], r.getPosition()[2]);

    //    RayF rayNEE = RayF(lDirection, position);
    //    rayNEE.AdvanceSelf(MathConstants::Epsilon);
    //    rayOut.ray = rayNEE;
    //    rayOut.tMin = 0.0f;
    //    rayOut.tMax =  lDistance - MathConstants::Epsilon;

    //    // Evaluate mat for this direction
    //    Vector3 reflectance = MGroup::Evaluate(// Input
    //                                           lDirection,
    //                                           -r.getDirection(),
    //                                           position,
    //                                           m,
    //                                           //
    //                                           surface,
    //                                           nullptr,
    //                                           // Constants
    //                                           gMatData,
    //                                           matIndex);

    //    // Incorporate for Radiance Factor
    //    auxOutNEE.radianceFactor *= reflectance / pdfLight;
    //    // Set Endpoint Index
    //    auxOutNEE.endPointIndex = lightIndex;

    //    // Write to global memory
    //    rayOut.Update(gOutRays, NEE_RAY_INDEX);
    //    gOutRayAux[NEE_RAY_INDEX] = auxOutNEE;
    //    gOutBoundKeys[NEE_RAY_INDEX] = matLight;
    //}
    //else
    //{
    //    // Generate Dummy Ray and Terminate
    //    RayReg rDummy = EMPTY_RAY_REGISTER;
    //    rDummy.Update(gOutRays, NEE_RAY_INDEX);
    //    gOutBoundKeys[NEE_RAY_INDEX] = HitKey::InvalidKey;
    //}    
}