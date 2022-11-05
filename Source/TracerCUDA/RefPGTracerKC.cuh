#pragma once

#include "RayAuxStruct.cuh"
#include "RefPGTracer.h"

#include "GPULightI.h"
#include "GPUMediumI.h"
#include "GPUDirectLightSamplerI.h"
#include "TracerConstants.h"
#include "WorkOutputWriter.cuh"

#include "RayLib/ColorConversion.h"

struct RPGTracerGlobalState
{
    // Output Samples
    CamSampleGMem<float>            gSamples;
    // Light Related
    const GPULightI**               gLightList;
    uint32_t                        totalLightCount;
    const GPUDirectLightSamplerI*   gLightSampler;
    // Medium Related
    const GPUMediumI**              mediumList;
    uint32_t                        totalMediumCount;
    // Render Resolution
    Vector2i                        resolution;
    ProjectionType                  projType;
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

__device__ inline
Vector2f ProjectSampleCoOctohedral(const Vector3& dir,
                                   const Vector2i& resolution)
{
    Vector3 dirZUp = Vector3(dir[2], dir[0], dir[1]);
    Vector2f st = Utility::DirectionToCocentricOctohedral(dirZUp);
    Vector2f sampleImgCoord = Vector2f(st[0] * static_cast<float>(resolution[0]),
                                       st[1] * static_cast<float>(resolution[1]));
    return sampleImgCoord;
}

__device__ inline
Vector2f ProjectSampleSpherical(const Vector3& dir,
                                const Vector2i& resolution)
{
    // Convert Y up from Z up
    Vector3 dirZUp = Vector3(dir[2], dir[0], dir[1]);
    Vector2f thetaPhi = Utility::CartesianToSphericalUnit(dirZUp);

    // Normalize to generate UV [0, 1]
    // theta range [-pi, pi]
    float u = (thetaPhi[0] + MathConstants::Pi) * 0.5f / MathConstants::Pi;
    // If we are at edge point (u == 1) make it zero since
    // piecewise constant function will not have that pdf (out of bounds)
    u = (u == 1.0f) ? 0.0f : u;
    // phi range [0, pi]
    float v = 1.0f - (thetaPhi[1] / MathConstants::Pi);
    // If (v == 1) then again pdf of would be out of bounds.
    // make it inbound
    v = (v == 1.0f) ? (v - MathConstants::SmallEpsilon) : v;

    // Check for numeric instability (just clamp the data)
    assert(u >= 0.0f && u < 1.0f);
    assert(v >= 0.0f && v < 1.0f);

    Vector2f sampleImgCoord = Vector2f(u * static_cast<float>(resolution[0]),
                                       v * static_cast<float>(resolution[1]));
    return sampleImgCoord;
}

template <class EGroup>
__device__ inline
void RPGTracerBoundaryWork(// Output
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
                           RPGTracerLocalState& localState,
                           RPGTracerGlobalState& renderState,
                           RNGeneratorGPUI& rng,
                          // Constants
                           const typename EGroup::GPUType& gLight)
{
    using GPUType = typename EGroup::GPUType;

    // No accum if the ray is camera ray
    if(aux.type == RayType::CAMERA_RAY) return;

    // Check Material Sample Strategy
    assert(maxOutRay == 0);

    const bool isPathRayAsMISRay = renderState.directLightMIS && (aux.type == RayType::PATH_RAY);
    const bool isSpecularPathRay = aux.type == RayType::SPECULAR_PATH_RAY;
    const bool isNeeRayNEEOn = renderState.nee && aux.type == RayType::NEE_RAY;
    const bool isPathRayNEEOff = (!renderState.nee) && (aux.type == RayType::PATH_RAY ||
                                                        aux.type == RayType::SPECULAR_PATH_RAY);
    // Always eval boundary mat if NEE is off
    // or NEE is on and hit endpoint and requested endpoint is same
    const GPULightI* requestedLight = (isNeeRayNEEOn) ? renderState.gLightList[aux.endpointIndex] : nullptr;
    const bool isCorrectNEERay = (isNeeRayNEEOn && (requestedLight->EndpointId() == gLight.EndpointId()));
    // If a path ray is hit when NEE is off, consider it as a camera ray
    bool isFirstDepthPathRay = (aux.depth == 2) && (aux.type == RayType::PATH_RAY);

    // Only treat path ray as MIS ray if it is not calculates the direct lighting
    float misWeight = 1.0f;
    if(isPathRayAsMISRay && !isFirstDepthPathRay)
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
    if(isPathRayNEEOff     || // We hit a light with a path ray while NEE is off
       isPathRayAsMISRay   || // We hit a light with a path ray while MIS option is enabled
       isCorrectNEERay     || // We hit the correct light as a NEE ray while NEE is on
       isFirstDepthPathRay || // These rays are "camera rays" of this surface
                              // which should not be culled when NEE is on
       isSpecularPathRay)     // We hit as spec ray which did not launched any NEE rays thus it should be hit
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
        float luminance = Utility::RGBToLuminance(total);
        AccumulateRaySample(renderState.gSamples, aux.sampleIndex, luminance);
    }
}

template <class MGroup>
__device__ inline
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
                       RPGTracerLocalState& localState,
                       RPGTracerGlobalState& renderState,
                       RNGeneratorGPUI& rng,
                       // Constants
                       const typename MGroup::Data& gMatData,
                       const HitKey::Type matIndex)
{
    static constexpr Vector3 ZERO_3 = Zero3;

    using SampleProjectionFunc = Vector2f(*)(const Vector3& dir,
                                             const Vector2i& resolution);

    static constexpr SampleProjectionFunc PROJECTION_FUNCTIONS[static_cast<int>(ProjectionType::END)] =
    {
        ProjectSampleSpherical,
        ProjectSampleCoOctohedral
    };

    // TODO: change this currently only first strategy is sampled
    static constexpr int PATH_RAY_INDEX = 0;
    static constexpr int NEE_RAY_INDEX  = 1;
    // Helper Class for better code readability
    OutputWriter<RayAuxPath> outputWriter(gOutBoundKeys,
                                          gOutRays,
                                          gOutRayAux,
                                          maxOutRay);

    // Inputs
    const bool isCameraRay = (aux.type == RayType::CAMERA_RAY);
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
        if(aux.type != RayType::CAMERA_RAY)
        {
            float luminance = Utility::RGBToLuminance(total);
            AccumulateRaySample(renderState.gSamples,
                                aux.sampleIndex,
                                luminance);
        }
    }

    // If this material does not require to have any samples just quit
    if(sampleCount == 0) return;

    bool shouldLaunchMISRay = false;
    // ===================================== //
    //              NEE PORTION              //
    // ===================================== //
    // Don't launch NEE if not requested
    // or material is highly specular
    if(renderState.nee && !isSpecularMat && !isCameraRay)
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
                              // dirac delta light (point light spot light etc.)
                              renderState.gLightList[lightIndex]->CanBeSampled() &&
                              // If current ray is camera ray we don't launch NEE ray anyway so
                              // this is kinda redundant
                              !isCameraRay);

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
        auxOut.type = (isSpecularMat) ? RayType::SPECULAR_PATH_RAY : RayType::PATH_RAY;
        // Save BxDF pdf if we hit a light
        // When we hit a light we will
        auxOut.prevPDF = pdfPath;
        auxOut.depth++;

        // Calculate the actual pixel id and add a sample
        // when the very first path ray is launched
        // Don't multiply with BxDF here
        if(isCameraRay)
        {
            // Set the actual image coordinate
            auto ProjectionFunction = PROJECTION_FUNCTIONS[static_cast<int>(renderState.projType)];
            Vector2f sampleImgCoord = ProjectionFunction(rayPath.getDirection(), renderState.resolution);
            //printf("NewCoord: %f, %f\n", sampleImgCoord[0], sampleImgCoord[1]);
            renderState.gSamples.gImgCoords[aux.sampleIndex] = sampleImgCoord;

            // Still divide with the path pdf since we sample the paths
            // using BxDF
            //auxOut.radianceFactor = Vector3(1.0f / pdfPath);
            auxOut.radianceFactor = Vector3(1.0f);
        }
        else
            auxOut.radianceFactor = pathRadianceFactor;


        // Write
        outputWriter.Write(PATH_RAY_INDEX, rayOut, auxOut);
    }
    // All Done!
}