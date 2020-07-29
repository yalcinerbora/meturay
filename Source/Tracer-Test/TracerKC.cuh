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
};

// No Local State
struct EmptyState {};

struct PathTracerLocal
{
    uint32_t materialSampleCount;
    bool     emissiveMaterial;
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
    if(neeMatch || aux.type == RayType::CAMERA_RAY)
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
    uint32_t sampleCount = maxOutRay;
    bool emissiveMaterial = gLocalState.emissiveMaterial;
    static int PATH_RAY_INDEX = 0;
    static int NEE_RAY_INDEX = 1;
    
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

    // If NEE ray hit to these material
    // which can not be sampled with NEE ray just skip
    if(aux.type == RayType::NEE_RAY)
    {
        // Write invalids for out rays
        for(uint32_t i = 0; i < sampleCount; i++)
            InvalidRayWrite(i);
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
    auxOut.radianceFactor = aux.radianceFactor * reflectance;
    // Check singularities
    auxOut.radianceFactor = (pdfPath == 0.0f) ? Zero3 : (auxOut.radianceFactor / pdfPath);

    // Change current medium of the ray
    auxOut.mediumIndex = static_cast<uint16_t>(outM.ID());

    //if(isnan(auxOut.radianceFactor[0]) ||
    //   isnan(auxOut.radianceFactor[1]) ||
    //   isnan(auxOut.radianceFactor[2]))
    //    printf("{%f, %f, %f} = {%f, %f, %f} * {%f, %f, %f} / %f\n",
    //           auxOut.radianceFactor[0],
    //           auxOut.radianceFactor[1],
    //           auxOut.radianceFactor[2],
    //           aux.radianceFactor[0],
    //           aux.radianceFactor[1],
    //           aux.radianceFactor[2],
    //           reflectance[0],
    //           reflectance[1],
    //           reflectance[2],
    //           pdfPath);

    // Check Russian Roulette
    float avgThroughput = auxOut.radianceFactor.Dot(Vector3f(0.333f));
    if(auxOut.depth <= gRenderState.rrStart ||
       !RussianRoulette(auxOut.radianceFactor, avgThroughput, rng))
    {
        // Write Ray
        RayReg rayOut;
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
    }

    // Dont launch NEE if not requested
    if(!gRenderState.nee) return;

    // NEE Ray Generation
    float pdfLight, lDistance;
    HitKey matLight;
    Vector3 lDirection;
    uint32_t lightIndex;
    reflectance = Zero3;
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
        // Evaluate mat for this direction
        reflectance = MGroup::Evaluate(// Input
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
    }

    // Do not waste sample if material does not reflect light
    // towards sampled position
    if(reflectance != Vector3(0.0f))
    {
        // Generate Ray
        RayF rayNEE = RayF(lDirection, position);
        rayNEE.AdvanceSelf(MathConstants::Epsilon);
        rayOut.ray = rayNEE;
        rayOut.tMin = 0.0f;
        rayOut.tMax = lDistance + MathConstants::Epsilon;
        // Write ray
        rayOut.Update(gOutRays, NEE_RAY_INDEX);

        // Calculate Radiance Factor
        auxOut.radianceFactor = aux.radianceFactor * reflectance / pdfLight;
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