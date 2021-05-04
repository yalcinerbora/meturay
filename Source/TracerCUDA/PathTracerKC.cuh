#pragma once

#include "RayAuxStruct.cuh"

#include "MaterialFunctions.cuh"
#include "ImageStructs.h"
#include "RayStructs.h"
#include "GPULightI.h"
#include "GPUDirectLightSamplerI.h"
#include "GPUMediumVacuum.cuh"
#include "GPUSurface.h"
#include "TracerFunctions.cuh"
#include "TracerConstants.h"

struct PathTracerGlobal
{
    // Output Image
    ImageGMem<Vector4> gImage;
    // Light Related
    const GPULightI**               lightList;
    uint32_t                        totalLightCount;
    const GPUDirectLightSamplerI*   lightSampler;
    // Medium Related
    const GPUMediumI**              mediumList;
    uint32_t                        totalMediumCount;

    // Options
    // Options for NEE
    bool                directLightMIS;
    bool                nee;
    int                 rrStart;
};

struct PathTracerLocal
{
    bool    emptyPrimitive;
};

template <class MGroup>
__device__
inline void PathTracerPathWork(// Output
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

    // Inputs
    // Current Ray
    const RayF& r = ray.ray;
    // Current Material Index
    HitKey::Type matIndex = HitKey::FetchIdPortion(matId);
    // Hit Position
    Vector3 position = r.AdvancedPos(ray.tMax);
    // Wi (direction is swapped as if it is coming out of the surface
    Vector3 wi = -(r.getDirection().Normalize());
    // Current ray's medium
    const GPUMediumI& m = *(gRenderState.mediumList[aux.mediumIndex]);

    // Check Material Sample Strategy
    const uint32_t sampleCount = maxOutRay;
    // Check Material's specularity;
    float specularity = MGroup::Specularity(surface, gMatData, matIndex);
    bool isSpecularMat = (specularity >= TracerConstants::SPECULAR_TRESHOLD);
   
    // TODO: change this currently only first strategy is sampled
    static constexpr int PATH_RAY_INDEX = 0;

    // Output image
    auto& img = gRenderState.gImage;

    // Invalid Ray Write Helper Function
    auto InvalidRayWrite = [&gOutRays, &gOutBoundKeys, &gOutRayAux, &sampleCount](int index)
    {
        assert(index < sampleCount);

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
        InvalidRayWrite(PATH_RAY_INDEX);
        return;
    }
    
    // Calculate Transmittance factor of the medium
    Vector3 transFactor = m.Transmittance(ray.tMax);
    Vector3 radianceFactor = aux.radianceFactor * transFactor;

    // Sample the emission if avail
    if(MGroup::IsEmissive(gMatData, matIndex))
    {
        Vector3 emission = MGroup::Emit(// Input
                                        wi,
                                        position,
                                        m,
                                        //
                                        surface,
                                        // Constants
                                        gMatData,
                                        matIndex);

        Vector3f total = emission * radianceFactor;
        ImageAccumulatePixel(img, aux.pixelIndex, Vector4f(total, 1.0f));
    }
        
    // If this material does not require to have any samples just quit
    // no need to sat any ray invalid since there wont be any allocated rays
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
    Vector3f pathRadianceFactor = radianceFactor * reflectance;
    // Check singularities
    pathRadianceFactor = (pdfPath == 0.0f) ? Zero3 : (pathRadianceFactor / pdfPath);
    
    // Check Russian Roulette
    float avgThroughput = pathRadianceFactor.Dot(Vector3f(0.333f));
    bool terminateRay = ((aux.depth > gRenderState.rrStart) &&
                         TracerFunctions::RussianRoulette(pathRadianceFactor, avgThroughput, rng));

    // Do not terminate rays ever for specular mats 
    if((!terminateRay || isSpecularMat) &&
        // Do not waste rays on zero radiance paths
        pathRadianceFactor != ZERO_3)
    {
        // Write Ray
        RayReg rayOut;
        rayOut.ray = rayPath;
        rayOut.tMin = 0.0f;
        rayOut.tMax = INFINITY;
        rayOut.Update(gOutRays, PATH_RAY_INDEX);

        // Write Aux
        RayAuxPath auxOut = aux;
        auxOut.mediumIndex = static_cast<uint16_t>(outM->GlobalIndex());
        auxOut.radianceFactor = pathRadianceFactor;
        auxOut.type = (isSpecularMat) ? RayType::SPECULAR_PATH_RAY : RayType::PATH_RAY;
        auxOut.depth++;
        gOutRayAux[PATH_RAY_INDEX] = auxOut;
    }
    else InvalidRayWrite(PATH_RAY_INDEX);
    // All Done!
}

template <class MGroup>
__device__
inline void PathTracerNEEWork(// Output
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
    static constexpr int NEE_RAY_INDEX = 0;
    static constexpr int PATH_RAY_INDEX = 0;
  
    // Inputs
    // Current Ray
    const RayF& r = ray.ray;
    // Current Material Index
    HitKey::Type matIndex = HitKey::FetchIdPortion(matId);
    // Hit Position
    Vector3 position = r.AdvancedPos(ray.tMax);
    // Wi (direction is swapped as if it is coming out of the surface
    Vector3 wi = -(r.getDirection().Normalize());
    // Current ray's medium
    const GPUMediumI& m = *(gRenderState.mediumList[aux.mediumIndex]);

    // Check Material Sample Strategy
    uint32_t sampleCount = maxOutRay;
    // Check Material's specularity;
    float specularity = MGroup::Specularity(surface, gMatData, matIndex);
    bool isSpecularMat = (specularity >= TracerConstants::SPECULAR_TRESHOLD);
 
    // Output image
    auto& img = gRenderState.gImage;

    // Invalid Ray Write Helper Function
    auto InvalidRayWrite = [&gOutRays, &gOutBoundKeys, &gOutRayAux, &sampleCount](int index)
    {
        assert(index < sampleCount);

        // Generate Dummy Ray and Terminate
        RayReg rDummy = EMPTY_RAY_REGISTER;
        rDummy.Update(gOutRays, index);
        gOutBoundKeys[index] = HitKey::InvalidKey;
        gOutRayAux[index].pixelIndex = UINT32_MAX;
     };

    // If NEE ray hits to this material
    // just skip since this is not a light material
    // Also skip if material is highly specular
    if(aux.type == RayType::NEE_RAY || isSpecularMat)
    {       
        InvalidRayWrite(NEE_RAY_INDEX);
        return;
    }
    
    // Calculate Transmittance factor of the medium
    Vector3 transFactor = m.Transmittance(ray.tMax);
    Vector3 radianceFactor = aux.radianceFactor * transFactor;

    // Material is not specular & we requested NEE ray
    // Generate NEE Ray    
    float pdfNEELight, lDistance;
    HitKey matLight;
    Vector3 lDirection;
    uint32_t lightIndex;
    Vector3f neeReflectance = Zero3;
    if(gRenderState.lightSampler->SampleLight(matLight,
                                              lightIndex,
                                              lDirection,
                                              lDistance,
                                              pdfNEELight,
                                              // Input
                                              position,
                                              rng))
    {
        // Evaluate mat for this direction
        neeReflectance = MGroup::Evaluate(// Input
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

    // Check if mis ray should be sampled
    bool launchedMISRay = (gRenderState.directLightMIS &&
                           // Check if light can be sampled (meaning it is not a
                           // dirac delta light (point light spot light etc.)
                           gRenderState.lightList[lightIndex]->CanBeSampled());

    // Sample Another ray for MIS (from BxDF)
    float pdfMIS = 0.0f;
    RayF rayMIS; const GPUMediumI* outMMIS;    
    Vector3f misReflectance = Zero3;
    if(launchedMISRay)
    {
        float pdfMISBxDF, pdfLightC, pdfLightM;
        misReflectance = MGroup::Sample(// Outputs
                                        rayMIS, pdfMISBxDF, outMMIS,
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

        // We are subsampling a single light pdf of BxDF also incorporate this
        gRenderState.lightSampler->Pdf(pdfLightM, pdfLightC,
                                       lightIndex, position, 
                                       rayMIS.getDirection());
        float pdfMIS = pdfLightC * pdfMISBxDF;
    }

    // Calculate PDF (or Combined PDF if MIS is enabled
    float neePDF = (launchedMISRay) 
            ? (pdfNEELight / TracerFunctions::PowerHeuristic(1, pdfNEELight, 1, pdfMIS))
            : pdfNEELight;

    // Do not waste a ray if material does not reflect
    // towards light's sampled position
    Vector3 neeRadianceFactor = radianceFactor * neeReflectance;
    neeRadianceFactor = (neePDF == 0.0f) ? Zero3 : (neeRadianceFactor / neePDF);
    if(neeRadianceFactor != ZERO_3)
    {
        // Generate & Write Ray
        RayF rayNEE = RayF(lDirection, position);
        rayNEE.AdvanceSelf(MathConstants::Epsilon);

        RayReg rayOut;
        rayOut.ray = rayNEE;
        rayOut.tMin = 0.0f;
        rayOut.tMax = lDistance;
        rayOut.Update(gOutRays, NEE_RAY_INDEX);

        RayAuxPath auxOut = aux;
        auxOut.radianceFactor = neeRadianceFactor;        
        auxOut.endPointIndex = lightIndex;
        auxOut.type = RayType::NEE_RAY;

        gOutRayAux[NEE_RAY_INDEX] = auxOut;
        gOutBoundKeys[NEE_RAY_INDEX] = matLight;
    }
    else InvalidRayWrite(NEE_RAY_INDEX);

    // Check MIS Ray return if not requested (since no ray is allocated for it)
    if(!gRenderState.directLightMIS) return;

    // Calculate Combined PDF
    float pdfCombined = pdfMIS / TracerFunctions::PowerHeuristic(1, pdfMIS, 1, pdfNEELight);
    Vector3 misRadianceFactor = radianceFactor * misReflectance;
    misRadianceFactor = (pdfCombined == 0.0f) ? Zero3 : (misRadianceFactor / pdfCombined);
    if(launchedMISRay &&
       misRadianceFactor != ZERO_3)
    {
        // Write Ray
        RayReg rayOut;
        rayOut.ray = rayMIS;
        rayOut.tMin = 0.0f;
        rayOut.tMax = INFINITY;
        rayOut.Update(gOutRays, MIS_RAY_INDEX);

        // Write Aux
        RayAuxPath auxOut = aux;
        auxOut.radianceFactor = misRadianceFactor;
        auxOut.endPointIndex = lightIndex;
        auxOut.type = RayType::NEE_RAY;

        gOutBoundKeys[MIS_RAY_INDEX] = matLight;
        gOutRayAux[MIS_RAY_INDEX] = auxOut;
    }
    else InvalidRayWrite(MIS_RAY_INDEX);

    // All Done!
}