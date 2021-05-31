#pragma once

#include "RayAuxStruct.cuh"

#include "GPULightI.h"
#include "GPUMediumI.h"
#include "GPUDirectLightSamplerI.h"

#include "TracerFunctions.cuh"
#include "TracerConstants.h"

//#include "MaterialFunctions.cuh"
//#include "ImageStructs.h"
//#include "RayStructs.h"
//#include "GPULightI.h"
//#include "GPUDirectLightSamplerI.h"
//#include "GPUMediumVacuum.cuh"
//#include "GPUSurface.h"


struct PathTracerGlobalState
{
    // Output Image
    ImageGMem<Vector4>              gImage;
    // Light Related
    const GPULightI**               lightList;
    uint32_t                        totalLightCount;
    const GPUDirectLightSamplerI*   lightSampler;
    // Medium Related
    const GPUMediumI**              mediumList;
    uint32_t                        totalMediumCount;

    // Options
    // Options for NEE
    bool                            directLightMIS;
    bool                            nee;
    int                             rrStart;
};

struct PathTracerLocalState
{
    bool    emptyPrimitive;
};

template <class MGroup>
__device__ __forceinline__
void PathTracerBoundaryWork(// Output
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
                            PathTracerLocalState& gLocalState,
                            PathTracerGlobalState& gRenderState,
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
__device__ __forceinline__
void PathTracerComboWork(// Output
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
                         PathTracerLocalState& gLocalState,
                         PathTracerGlobalState& gRenderState,
                         RandomGPU& rng,
                         // Constants
                         const typename MGroup::Data& gMatData,
                         const HitKey matId,
                         const PrimitiveId primId)
{
    static constexpr Vector3 ZERO_3 = Zero3;

    // TODO: change this currently only first strategy is sampled
    static constexpr int PATH_RAY_INDEX = 0;
    static constexpr int NEE_RAY_INDEX = 1;
    static constexpr int MIS_RAY_INDEX = 2;

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
        // Write invalids for out rays
        for(uint32_t i = 0; i < sampleCount; i++)
            InvalidRayWrite(i);
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
        // Output image
        auto& img = gRenderState.gImage;
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

    // Dont launch NEE if not requested
    // or material is highly specula
    if(!gRenderState.nee) return;

    // Renderer requested a NEE Ray but material is highly specular
    // Check if nee is requested
    if(isSpecularMat && maxOutRay == 1)
        return;
    else if(isSpecularMat)
    {
        // Write invalid rays then return
        InvalidRayWrite(NEE_RAY_INDEX);
        if(gRenderState.directLightMIS)
            InvalidRayWrite(MIS_RAY_INDEX);
        return;
    }

    // Material is not specular & tracer requested a NEE ray
    // Generate a NEE Ray
    float pdfLight, lDistance;
    HitKey matLight;
    Vector3 lDirection;
    uint32_t lightIndex;
    Vector3f neeReflectance = Zero3;
    if(gRenderState.lightSampler->SampleLight(matLight,
                                              lightIndex,
                                              lDirection,
                                              lDistance,
                                              pdfLight,
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

    float pdfNEE = pdfLight;
    if(launchedMISRay)
    {
        float pdfBxDF = MGroup::Pdf(lDirection,
                                    wi,
                                    position,
                                    m,
                                    //
                                    surface,
                                    gMatData,
                                    matIndex);

        pdfNEE /= TracerFunctions::PowerHeuristic(1, pdfLight, 1, pdfBxDF);

        // PDF can become NaN if both BxDF pdf and light pdf is both zero 
        // (meaning both sampling schemes does not cover this direction)
        if(isnan(pdfNEE)) pdfNEE = 0.0f;
    }

    // Do not waste a ray if material does not reflect
    // towards light's sampled position
    Vector3 neeRadianceFactor = radianceFactor * neeReflectance;
    neeRadianceFactor = (pdfNEE == 0.0f) ? Zero3 : (neeRadianceFactor / pdfNEE);
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

    // Sample Another ray for MIS (from BxDF)
    float pdfMIS = 0.0f;
    RayF rayMIS; const GPUMediumI* outMMIS;    
    Vector3f misReflectance = Zero3;
    if(launchedMISRay)
    {
        misReflectance = MGroup::Sample(// Outputs
                                        rayMIS, pdfMIS, outMMIS,
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

        // Find out the pdf of the light
        float pdfLightM, pdfLightC;
        gRenderState.lightSampler->Pdf(pdfLightM, pdfLightC,
                                       lightIndex, position,
                                       rayMIS.getDirection());
        // We are subsampling (discretely sampling) a single light 
        // pdf of BxDF should also incorporate this
        pdfMIS *= pdfLightM;

        pdfMIS /= TracerFunctions::PowerHeuristic(1, pdfMIS, 1, pdfLightC * pdfLightM);

        // PDF can become NaN if both BxDF pdf and light pdf is both zero 
        // (meaning both sampling schemes does not cover this direction)
        if(isnan(pdfMIS)) pdfMIS = 0.0f;
    }

    // Calculate Combined PDF
    Vector3 misRadianceFactor = radianceFactor * misReflectance;
    misRadianceFactor = (pdfMIS == 0.0f) ? Zero3 : (misRadianceFactor / pdfMIS);
    if(launchedMISRay && misRadianceFactor != ZERO_3)
    {
        // Write Ray
        RayReg rayOut;
        rayOut.ray = rayMIS;
        rayOut.tMin = 0.0f;
        rayOut.tMax = INFINITY;
        rayOut.Update(gOutRays, MIS_RAY_INDEX);

        // Write Aux
        RayAuxPath auxOut = aux;
        auxOut.mediumIndex = static_cast<uint16_t>(outMMIS->GlobalIndex());
        auxOut.radianceFactor = misRadianceFactor;
        auxOut.endPointIndex = lightIndex;
        auxOut.type = RayType::NEE_RAY;

        gOutBoundKeys[MIS_RAY_INDEX] = matLight;
        gOutRayAux[MIS_RAY_INDEX] = auxOut;
    }
    else InvalidRayWrite(MIS_RAY_INDEX);

    // All Done!
}

template <class MGroup>
__device__ __forceinline__
void PathTracerPathWork(// Output
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
                        PathTracerLocalState& gLocalState,
                        PathTracerGlobalState& gRenderState,
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
__device__ __forceinline__
void PathTracerNEEWork(// Output
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
                       PathTracerLocalState& gLocalState,
                       PathTracerGlobalState& gRenderState,
                       RandomGPU& rng,
                       // Constants
                       const typename MGroup::Data& gMatData,
                       const HitKey matId,
                       const PrimitiveId primId)
{
    static constexpr Vector3 ZERO_3 = Zero3;

    static constexpr int NEE_RAY_INDEX = 0;

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
    // Also skip is material is specular since
    // you cant sample specular material
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
    float pdfLight, lDistance;
    HitKey matLight;
    Vector3 lDirection;
    uint32_t lightIndex;
    Vector3f neeReflectance = Zero3;
    if(gRenderState.lightSampler->SampleLight(matLight,
                                              lightIndex,
                                              lDirection,
                                              lDistance,
                                              pdfLight,
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

    float pdfNEE = pdfLight;
    if(launchedMISRay)
    {
        float pdfBxDF = MGroup::Pdf(lDirection,
                                    wi,
                                    position,
                                    m,
                                    //
                                    surface,
                                    gMatData,
                                    matIndex);

        pdfNEE /= TracerFunctions::PowerHeuristic(1, pdfLight, 1, pdfBxDF);
    }

    // Do not waste a ray if material does not reflect
    // towards light's sampled position
    Vector3 neeRadianceFactor = radianceFactor * neeReflectance;
    neeRadianceFactor = (pdfNEE == 0.0f) ? Zero3 : (neeRadianceFactor / pdfNEE);
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
}

template <class MGroup>
__device__ __forceinline__
void PathTracerMISWork(// Output
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
                       PathTracerLocalState& gLocalState,
                       PathTracerGlobalState& gRenderState,
                       RandomGPU& rng,
                       // Constants
                       const typename MGroup::Data& gMatData,
                       const HitKey matId,
                       const PrimitiveId primId)
{
    //static constexpr Vector3 ZERO_3 = Zero3;
    //static constexpr int MIS_RAY_INDEX = 0;

    //uint32_t lightIndex;
    //HitKey matLight;

    //// Inputs
    //// Current Ray
    //const RayF& r = ray.ray;
    //// Current Material Index
    //HitKey::Type matIndex = HitKey::FetchIdPortion(matId);
    //// Hit Position
    //Vector3 position = r.AdvancedPos(ray.tMax);
    //// Wi (direction is swapped as if it is coming out of the surface
    //Vector3 wi = -(r.getDirection().Normalize());
    //// Current ray's medium
    //const GPUMediumI& m = *(gRenderState.mediumList[aux.mediumIndex]);

    //// Check Material Sample Strategy
    //const uint32_t sampleCount = maxOutRay;
    //// Check Material's specularity;
    //float specularity = MGroup::Specularity(surface, gMatData, matIndex);
    //bool isSpecularMat = (specularity >= TracerConstants::SPECULAR_TRESHOLD);
    //  
    //// Invalid Ray Write Helper Function
    //auto InvalidRayWrite = [&gOutRays, &gOutBoundKeys, &gOutRayAux, &sampleCount](int index)
    //{
    //    assert(index < sampleCount);

    //    // Generate Dummy Ray and Terminate
    //    RayReg rDummy = EMPTY_RAY_REGISTER;
    //    rDummy.Update(gOutRays, index);
    //    gOutBoundKeys[index] = HitKey::InvalidKey;
    //    gOutRayAux[index].pixelIndex = UINT32_MAX;
    // };

    //// If NEE ray hits to this material
    //// just skip since this is not a light material
    //// Also skip is material is specular since
    //// you cant sample specular material
    //if(aux.type == RayType::NEE_RAY || isSpecularMat)
    //{
    //    InvalidRayWrite(MIS_RAY_INDEX);
    //    return;
    //}
    //
    //// Calculate Transmittance factor of the medium
    //Vector3 transFactor = m.Transmittance(ray.tMax);
    //Vector3 radianceFactor = aux.radianceFactor * transFactor;

    //// Check if mis ray should be sampled
    //bool launchedMISRay = (gRenderState.directLightMIS &&
    //                       // Check if light can be sampled (meaning it is not a
    //                       // dirac delta light (point light spot light etc.)
    //                       gRenderState.lightList[lightIndex]->CanBeSampled());

    //// Sample Another ray for MIS (from BxDF)
    //float pdfMIS = 0.0f;
    //RayF rayMIS; const GPUMediumI* outMMIS;    
    //Vector3f misReflectance = Zero3;
    //if(launchedMISRay)
    //{
    //    misReflectance = MGroup::Sample(// Outputs
    //                                    rayMIS, pdfMIS, outMMIS,
    //                                    // Inputs
    //                                    wi,
    //                                    position,
    //                                    m,
    //                                    //
    //                                    surface,
    //                                    // I-O
    //                                    rng,
    //                                    // Constants
    //                                    gMatData,
    //                                    matIndex,
    //                                    0);

    //    // We are subsampling a single light pdf of BxDF also incorporate this
    //    float pdfLightM, pdfLightC;
    //    gRenderState.lightSampler->Pdf(pdfLightM, pdfLightC,
    //                                   lightIndex, position,
    //                                   rayMIS.getDirection());

    //    pdfMIS /= TracerFunctions::PowerHeuristic(1, pdfMIS, 1, pdfLightC);
    //    pdfMIS /= pdfLightM;
    //}

    //// Calculate Combined PDF
    //Vector3 misRadianceFactor = radianceFactor * misReflectance;
    //misRadianceFactor = (pdfMIS == 0.0f) ? Zero3 : (misRadianceFactor / pdfMIS);
    //if(launchedMISRay && misRadianceFactor != ZERO_3)
    //{
    //    // Write Ray
    //    RayReg rayOut;
    //    rayOut.ray = rayMIS;
    //    rayOut.tMin = 0.0f;
    //    rayOut.tMax = INFINITY;
    //    rayOut.Update(gOutRays, MIS_RAY_INDEX);

    //    // Write Aux
    //    RayAuxPath auxOut = aux;
    //    auxOut.radianceFactor = misRadianceFactor;
    //    auxOut.endPointIndex = lightIndex;
    //    auxOut.type = RayType::NEE_RAY;

    //    gOutBoundKeys[MIS_RAY_INDEX] = matLight;
    //    gOutRayAux[MIS_RAY_INDEX] = auxOut;
    //}
    //else InvalidRayWrite(MIS_RAY_INDEX);
}