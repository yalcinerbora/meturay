#pragma once

#include "RayAuxStruct.cuh"

#include "GPULightI.h"
#include "GPUMediumI.h"
#include "GPUDirectLightSamplerI.h"

#include "TracerFunctions.cuh"
#include "TracerConstants.h"
#include "GPUMetaSurfaceGenerator.h"
#include "RNGIndependent.cuh"
#include "ImageFunctions.cuh"

#include "WorkOutputWriter.cuh"

struct PathTracerGlobalState
{
    // Output Samples
    CamSampleGMem<Vector4f>         gSamples;
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
    // Russian Roulette
    int                             rrStart;
};

struct PathTracerLocalState
{
    bool    emptyPrimitive;
};

__global__
static void KCPathTracerMegaKernel(// Output
                                   HitKey* gOutBoundKeys,
                                   RayGMem* gOutRays,
                                   RayAuxPath* gOutRayAux,
                                   const uint8_t maxOutRay,
                                   // Inputs
                                   const RayId* gRayIds,
                                   const RayAuxPath* gInRayAux,
                                   GPUMetaSurfaceGeneratorGroup surfaceGenerator,
                                   // I-O
                                   PathTracerGlobalState renderState,
                                   RNGeneratorGPUI** gRNGs,
                                   // MetaSurfaceGenerator

                                   // Constants
                                   const uint32_t rayCount,
                                   const GPUTransformI* const* gTransforms)
{
    auto& rng = RNGAccessor::Acquire<RNGIndependentGPU>(gRNGs, LINEAR_GLOBAL_ID);
    for(uint32_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
        globalId < rayCount; globalId += blockDim.x * gridDim.x)
    {
        const RayId rayId = gRayIds[globalId];
        //const HitKey hitKey = gMatIds[globalId];
        //const HitKey::Type batchId = HitKey::FetchBatchPortion(hitKey);

        //// If Invalid ray, completely skip
        //if(HitKey::FetchBatchPortion(hitKey) == HitKey::NullBatch)
        //    continue;

        // Load Input to Registers
        const RayReg ray = surfaceGenerator.Ray(rayId);
        const RayAuxPath aux = gInRayAux[rayId];
        const PrimitiveId primitiveId = surfaceGenerator.PrimId(rayId);
        //const TransformId transformId = gTransformIds[rayId];

        GPUMetaSurface metaSurface = surfaceGenerator.AcquireWork(rayId);
        // Normal Path Tracing
        if(metaSurface.IsLight())
        {
            assert(maxOutRay == 0);
            // Special Case: Null Light just skip
            if(metaSurface.IsNullLight()) continue;
            // Skip if primitiveId is invalid only if the light is
            // primitive backed.
            // This happens when NEE generates a ray with a
            // predefined workId (which did invoke this thread)
            // However the light is missed somehow
            // (planar rays, numeric instability etc.)
            // Because of that primitive id did not get populated
            // Skip this ray
            if(metaSurface.IsPrimitiveBackedLight() && primitiveId >= INVALID_PRIMITIVE_ID)
                continue;

            const bool isPathRayAsMISRay = renderState.directLightMIS && (aux.type == RayType::PATH_RAY);
            const bool isCameraRay = aux.type == RayType::CAMERA_RAY;
            const bool isSpecularPathRay = aux.type == RayType::SPECULAR_PATH_RAY;
            const bool isNeeRayNEEOn = renderState.nee && aux.type == RayType::NEE_RAY;
            const bool isPathRayNEEOff = (!renderState.nee) && (aux.type == RayType::PATH_RAY ||
                                                                aux.type == RayType::SPECULAR_PATH_RAY);
            // Always eval boundary mat if NEE is off
            // or NEE is on and hit endpoint and requested endpoint is same
            const GPULightI* requestedLight = (isNeeRayNEEOn) ? renderState.gLightList[aux.endpointIndex] : nullptr;
            const bool isCorrectNEERay = (isNeeRayNEEOn && (requestedLight->EndpointId() == metaSurface.EndpointId()));

            float misWeight = 1.0f;
            if(isPathRayAsMISRay)
            {
                Vector3 position = metaSurface.WorldPosition();
                Vector3 direction = ray.ray.getDirection().Normalize();

                // Find out the pdf of the light
                float pdfLightM, pdfLightC;
                renderState.gLightSampler->Pdf(pdfLightM, pdfLightC,
                                               //
                                               metaSurface.GlobalLightIndex(),
                                               ray.tMax,
                                               position,
                                               direction,
                                               metaSurface.WorldToTangent());

                // We are sub-sampling (discretely sampling) a single light
                // pdf of BxDF should also incorporate this
                float bxdfPDF = aux.prevPDF;
                misWeight = TracerFunctions::PowerHeuristic(1, bxdfPDF,
                                                            1, pdfLightC * pdfLightM);
            }

            // Accumulate Light if
            if(isPathRayNEEOff ||   // We hit a light with a path ray while NEE is off
               isPathRayAsMISRay || // We hit a light with a path ray while MIS option is enabled
               isCorrectNEERay ||   // We hit the correct light as a NEE ray while NEE is on
               isCameraRay ||       // We hit as a camera ray which should not be culled when NEE is on
               isSpecularPathRay)   // We hit as spec ray which did not launched any NEE rays thus it should contribute
            {
                // Data Fetch
                const RayF& r = ray.ray;
                Vector3 position = metaSurface.WorldPosition();
                const GPUMediumI& m = *(renderState.mediumList[aux.mediumIndex]);

                // Calculate Transmittance factor of the medium
                // And reduce the radiance wrt the medium transmittance
                Vector3 transFactor = m.Transmittance(ray.tMax);
                Vector3 radianceFactor = aux.radianceFactor * transFactor;

                Vector3 emission = metaSurface.Emit(// Input
                                                    -r.getDirection(),
                                                    m);
                // And accumulate pixel// and add as a sample
                Vector3f total = emission * radianceFactor;
                // Incorporate MIS weight if applicable
                // if path ray hits a light misWeight is calculated
                // else misWeight is 1.0f
                total *= misWeight;
                // Accumulate the pixel
                AccumulateRaySample(renderState.gSamples,
                                    aux.sampleIndex,
                                    Vector4f(total, 1.0f));
            }
        }
        else
        {
            static constexpr int PATH_RAY_INDEX = 0;
            static constexpr int NEE_RAY_INDEX = 1;
            // Helper Class for better code readability
            OutputWriter<RayAuxPath> outputWriter(gOutBoundKeys + maxOutRay * globalId,
                                                  gOutRays + maxOutRay * globalId,
                                                  gOutRayAux + maxOutRay * globalId,
                                                  maxOutRay);

            static constexpr Vector3 ZERO_3 = Zero3;
            // Inputs
            // Current Ray
            const RayF& r = ray.ray;
            // Hit Position
            Vector3 position = metaSurface.WorldPosition();
            // Wi (direction is swapped as if it is coming out of the surface)
            Vector3 wi = -(r.getDirection().Normalize());
            // Current ray's medium
            const GPUMediumI& m = *(renderState.mediumList[aux.mediumIndex]);

            // Check Material Sample Strategy
            uint32_t sampleCount = maxOutRay;
            // Check Material's specularity;
            float specularity = metaSurface.Specularity();
            bool isSpecularMat = (specularity >= TracerConstants::SPECULAR_THRESHOLD);

            // If NEE ray hits to this material
            // just skip since this is not a light material
            if(aux.type == RayType::NEE_RAY) continue;

            // Calculate Transmittance factor of the medium
            // And reduce the radiance wrt the medium transmittance
            Vector3 transFactor = m.Transmittance(ray.tMax);
            Vector3 radianceFactor = aux.radianceFactor * transFactor;

            // Sample the emission if avail
            if(metaSurface.IsEmissive())
            {
                Vector3 emission = metaSurface.Emit(// Input
                                                wi,
                                                m);
                Vector3f total = emission * radianceFactor;
                AccumulateRaySample(renderState.gSamples,
                                    aux.sampleIndex,
                                    Vector4f(total, 1.0f));
            }

            // If this material does not require to have any samples just quit
            if(sampleCount == 0) continue;

            bool shouldLaunchMISRay = false;
            // ===================================== //
            //              NEE PORTION              //
            // ===================================== //
            // Don't launch NEE if not requested
            // or material is highly specular
            if(renderState.nee && !isSpecularMat)
            {
                float pdfLight, lDistance;
                HitKey matLight;
                Vector3 lDirection;
                uint32_t lightIndex;
                Vector3f neeReflectance = Zero3;
                if(renderState.gLightSampler->SampleLight(matLight,
                                                          lightIndex,
                                                          lDirection,
                                                          lDistance,
                                                          pdfLight,
                                                          // Input
                                                          position,
                                                          rng))
                {
                    // Evaluate mat for this direction
                    neeReflectance = metaSurface.Evaluate(// Input
                                                      lDirection,
                                                      wi,
                                                      m);
                }

                // Check if mis ray should be sampled
                shouldLaunchMISRay = (renderState.directLightMIS &&
                                      // Check if light can be sampled (meaning it is not a
                                      // Dirac delta light (point light spot light etc.)
                                      renderState.gLightList[lightIndex]->CanBeSampled());

                float pdfNEE = pdfLight;
                // Weight the NEE if using MIS
                if(shouldLaunchMISRay)
                {
                    float pdfBxDF = metaSurface.Pdf(lDirection,
                                                    wi,
                                                    m);

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
                    // Generate Ray
                    RayF rayNEE = RayF(lDirection, position);
                    RayReg rayOut;
                    rayOut.ray = rayNEE.Nudge(metaSurface.WorldGeoNormal(), 0.0f);
                    rayOut.tMin = 0.0f;
                    rayOut.tMax = lDistance;
                    // Aux
                    RayAuxPath auxOut = aux;
                    auxOut.radianceFactor = neeRadianceFactor;
                    auxOut.endpointIndex = lightIndex;
                    auxOut.type = RayType::NEE_RAY;
                    auxOut.prevPDF = NAN;
                    outputWriter.Write(NEE_RAY_INDEX, rayOut, auxOut, matLight);
                }
            }

            // ==================================== //
            //             BxDF PORTION             //
            // ==================================== //
            // Sample a path from material
            RayF rayPath; float pdfPath; const GPUMediumI* outM;
            Vector3 reflectance = metaSurface.Sample(// Outputs
                                                     rayPath, pdfPath, outM,
                                                     // Inputs
                                                     wi,
                                                     m,
                                                     rng);
            // Factor the radiance of the surface
            Vector3f pathRadianceFactor = radianceFactor * reflectance;
            // Check singularities
            pathRadianceFactor = (pdfPath == 0.0f) ? Zero3 : (pathRadianceFactor / pdfPath);

            // Check Russian Roulette
            float avgThroughput = pathRadianceFactor.Dot(Vector3f(0.333f));
            bool terminateRay = ((aux.depth > renderState.rrStart) &&
                                 TracerFunctions::RussianRoulette(pathRadianceFactor, avgThroughput, rng));

            // Do not terminate rays ever for specular mats
            if((!terminateRay || isSpecularMat) &&
                // Do not waste rays on zero radiance paths
               pathRadianceFactor != ZERO_3)
            {
                // Ray
                RayReg rayOut;
                rayOut.ray = rayPath;
                rayOut.tMin = 0.0f;
                rayOut.tMax = FLT_MAX;
                // Aux
                RayAuxPath auxOut = aux;
                auxOut.mediumIndex = static_cast<uint16_t>(outM->GlobalIndex());
                auxOut.radianceFactor = pathRadianceFactor;
                auxOut.type = (isSpecularMat) ? RayType::SPECULAR_PATH_RAY : RayType::PATH_RAY;
                // Save BxDF pdf if we hit a light
                // When we hit a light we will
                auxOut.prevPDF = pdfPath;
                auxOut.depth++;
                // Write
                outputWriter.Write(PATH_RAY_INDEX, rayOut, auxOut);
            }
            // All Done!
        }
    }
}

template <class EGroup>
__device__ inline
void PathTracerBoundaryWork(// Output
                            HitKey* gOutBoundKeys,
                            RayGMem* gOutRays,
                            RayAuxPath* gOutRayAux,
                            const uint32_t maxOutRay,
                            // Input as registers
                            const RayReg& ray,
                            const RayAuxPath& aux,
                            const typename EGroup::Surface& surface,
                            const RayId rayId,
                            // I-O
                            PathTracerLocalState& localState,
                            PathTracerGlobalState& renderState,
                            RNGeneratorGPUI& rng,
                            // Constants
                            const typename EGroup::GPUType& gLight)
{
    using GPUType = typename EGroup::GPUType;

    // Check Material Sample Strategy
    assert(maxOutRay == 0);

    const bool isPathRayAsMISRay = renderState.directLightMIS && (aux.type == RayType::PATH_RAY);
    const bool isCameraRay = aux.type == RayType::CAMERA_RAY;
    const bool isSpecularPathRay = aux.type == RayType::SPECULAR_PATH_RAY;
    const bool isNeeRayNEEOn = renderState.nee && aux.type == RayType::NEE_RAY;
    const bool isPathRayNEEOff = (!renderState.nee) && (aux.type == RayType::PATH_RAY ||
                                                        aux.type == RayType::SPECULAR_PATH_RAY);
    // Always eval boundary mat if NEE is off
    // or NEE is on and hit endpoint and requested endpoint is same
    const GPULightI* requestedLight = (isNeeRayNEEOn) ? renderState.gLightList[aux.endpointIndex] : nullptr;
    const bool isCorrectNEERay = (isNeeRayNEEOn && (requestedLight->EndpointId() == gLight.EndpointId()));

    float misWeight = 1.0f;
    if(isPathRayAsMISRay)
    {
        Vector3 position = surface.WorldPosition();
        Vector3 direction = ray.ray.getDirection().Normalize();

        // Find out the pdf of the light
        float pdfLightM, pdfLightC;
        renderState.gLightSampler->Pdf(pdfLightM, pdfLightC,
                                       //
                                       gLight.GlobalLightIndex(),
                                       ray.tMax,
                                       position,
                                       direction,
                                       surface.worldToTangent);

        // We are sub-sampling (discretely sampling) a single light
        // pdf of BxDF should also incorporate this
        float bxdfPDF = aux.prevPDF;
        misWeight = TracerFunctions::PowerHeuristic(1, bxdfPDF,
                                                    1, pdfLightC * pdfLightM);
    }

    // Accumulate Light if
    if(isPathRayNEEOff   || // We hit a light with a path ray while NEE is off
       isPathRayAsMISRay || // We hit a light with a path ray while MIS option is enabled
       isCorrectNEERay   || // We hit the correct light as a NEE ray while NEE is on
       isCameraRay       || // We hit as a camera ray which should not be culled when NEE is on
       isSpecularPathRay)   // We hit as spec ray which did not launched any NEE rays thus it should contribute
    {
        // Data Fetch
        const RayF& r = ray.ray;
        Vector3 position = surface.WorldPosition();
        const GPUMediumI& m = *(renderState.mediumList[aux.mediumIndex]);

        // Calculate Transmittance factor of the medium
        // And reduce the radiance wrt the medium transmittance
        Vector3 transFactor = m.Transmittance(ray.tMax);
        Vector3 radianceFactor = aux.radianceFactor * transFactor;

        Vector3 emission = gLight.Emit(// Input
                                       -r.getDirection(),
                                       position,
                                       surface);

        // And accumulate pixel// and add as a sample
        Vector3f total =  emission * radianceFactor;
        // Incorporate MIS weight if applicable
        // if path ray hits a light misWeight is calculated
        // else misWeight is 1.0f
        total *= misWeight;
        // Accumulate the pixel
        AccumulateRaySample(renderState.gSamples,
                            aux.sampleIndex,
                            Vector4f(total, 1.0f));
    }
}

template <class MGroup>
__device__ inline
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
                        PathTracerLocalState& localState,
                        PathTracerGlobalState& renderState,
                        RNGeneratorGPUI& rng,
                        // Constants
                        const typename MGroup::Data& gMatData,
                        const HitKey::Type matIndex)
{
    static constexpr Vector3 ZERO_3 = Zero3;

    // TODO: change this currently only first strategy is sampled
    static constexpr int PATH_RAY_INDEX = 0;
    static constexpr int NEE_RAY_INDEX  = 1;
    // Helper Class for better code readability
    OutputWriter<RayAuxPath> outputWriter(gOutBoundKeys,
                                          gOutRays,
                                          gOutRayAux,
                                          maxOutRay);

    // Inputs
    // Current Ray
    const RayF& r = ray.ray;
    // Hit Position
    Vector3 position = surface.WorldPosition();
    // Wi (direction is swapped as if it is coming out of the surface)
    Vector3 wi = -(r.getDirection().Normalize());
    // Current ray's medium
    const GPUMediumI& m = *(renderState.mediumList[aux.mediumIndex]);

    // Check Material Sample Strategy
    uint32_t sampleCount = maxOutRay;
    // Check Material's specularity;
    float specularity = MGroup::Specularity(surface, gMatData, matIndex);
    bool isSpecularMat = (specularity >= TracerConstants::SPECULAR_THRESHOLD);

    // If NEE ray hits to this material
    // just skip since this is not a light material
    if(aux.type == RayType::NEE_RAY) return;

    // Calculate Transmittance factor of the medium
    // And reduce the radiance wrt the medium transmittance
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
        AccumulateRaySample(renderState.gSamples,
                            aux.sampleIndex,
                            Vector4f(total, 1.0f));
    }

    // If this material does not require to have any samples just quit
    if(sampleCount == 0) return;

    bool shouldLaunchMISRay = false;
    // ===================================== //
    //              NEE PORTION              //
    // ===================================== //
    // Don't launch NEE if not requested
    // or material is highly specular
    if(renderState.nee && !isSpecularMat)
    {
        float pdfLight, lDistance;
        HitKey matLight;
        Vector3 lDirection;
        uint32_t lightIndex;
        Vector3f neeReflectance = Zero3;
        if(renderState.gLightSampler->SampleLight(matLight,
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
        shouldLaunchMISRay = (renderState.directLightMIS &&
                              // Check if light can be sampled (meaning it is not a
                              // Dirac delta light (point light spot light etc.)
                              renderState.gLightList[lightIndex]->CanBeSampled());

        float pdfNEE = pdfLight;
        // Weight the NEE if using MIS
        if(shouldLaunchMISRay)
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
            // Generate Ray
            RayF rayNEE = RayF(lDirection, position);
            RayReg rayOut;
            rayOut.ray = rayNEE.Nudge(surface.WorldGeoNormal(), surface.curvatureOffset);
            rayOut.tMin = 0.0f;
            rayOut.tMax = lDistance;
            // Aux
            RayAuxPath auxOut = aux;
            auxOut.radianceFactor = neeRadianceFactor;
            auxOut.endpointIndex = lightIndex;
            auxOut.type = RayType::NEE_RAY;
            auxOut.prevPDF = NAN;
            outputWriter.Write(NEE_RAY_INDEX, rayOut, auxOut, matLight);
        }
    }

    // ==================================== //
    //             BxDF PORTION             //
    // ==================================== //
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
    bool terminateRay = ((aux.depth > renderState.rrStart) &&
                         TracerFunctions::RussianRoulette(pathRadianceFactor, avgThroughput, rng));

    // Do not terminate rays ever for specular mats
    if((!terminateRay || isSpecularMat) &&
        // Do not waste rays on zero radiance paths
        pathRadianceFactor != ZERO_3)
    {
        // Ray
        RayReg rayOut;
        rayOut.ray = rayPath;
        rayOut.tMin = 0.0f;
        rayOut.tMax = FLT_MAX;
        // Aux
        RayAuxPath auxOut = aux;
        auxOut.mediumIndex = static_cast<uint16_t>(outM->GlobalIndex());
        auxOut.radianceFactor = pathRadianceFactor;
        auxOut.type = (isSpecularMat) ? RayType::SPECULAR_PATH_RAY : RayType::PATH_RAY;
        // Save BxDF pdf if we hit a light
        // When we hit a light we will
        auxOut.prevPDF = pdfPath;
        auxOut.depth++;
        // Write
        outputWriter.Write(PATH_RAY_INDEX, rayOut, auxOut);
    }
    // All Done!
}