#pragma once

#include "RayAuxStruct.cuh"

#include "GPULightI.h"
#include "GPUMediumI.h"
#include "GPUDirectLightSamplerI.h"
#include "TracerConstants.h"
#include "WorkOutputWriter.cuh"

#include "RayLib/ColorConversion.h"

struct RPGTracerGlobalState
{
    // Output Image
    ImageGMem<float>                gImage;
    // Light Related
    const GPULightI**               gLightList;
    uint32_t                        totalLightCount;
    const GPUDirectLightSamplerI*   gLightSampler;
    // Medium Related
    const GPUMediumI**              mediumList;
    uint32_t                        totalMediumCount;
    // Render Resolution
    Vector2i                        resolution;
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

__device__ __forceinline__
uint32_t CalculateSphericalPixelId(const Vector3& dir,
                                   const Vector2i& resolution)
{
    // Convert Y up from Z up
    Vector3 dirZup = Vector3(dir[2], dir[0], dir[1]);
    // Convert to Spherical Coordinates
    Vector2f thetaPhi = Utility::CartesianToSphericalUnit(dirZup);
    // Normalize to generate UV [0, 1]
    // tetha range [-pi, pi]
    float u = (thetaPhi[0] + MathConstants::Pi) * 0.5f / MathConstants::Pi;
    // phi range [0, pi]
    float v = 1.0f - (thetaPhi[1] / MathConstants::Pi);

    // Check for numeric unstaibility (just clamp the data)
    u = HybridFuncs::Clamp(u, 0.0f, 1.0f - MathConstants::Epsilon);
    v = HybridFuncs::Clamp(v, 0.0f, 1.0f - MathConstants::Epsilon);
    assert(u >= 0.0f && u < 1.0f);
    assert(v >= 0.0f && v < 1.0f);

    Vector2i pixelId2D = Vector2i(u * resolution[0],
                                  v * resolution[1]);
    uint32_t pixel1D = pixelId2D[1] * resolution[0] + pixelId2D[0];

    if(pixel1D >= resolution[0] * resolution[1])
    {
        printf("pixel out of range :\n"
               "uv   : [%f, %f]\n"
               "pix  : [%d, %d]\n"
               "pix1D: %u\n",
               u, v,
               pixelId2D[0], pixelId2D[1],
               pixel1D);
    }

    return pixel1D;
}

template <class EGroup>
__device__ __forceinline__
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
                           RandomGPU& rng,
                          // Constants
                           const typename EGroup::GPUType& gLight)
{
    using GPUType = typename EGroup::GPUType;

    // No accum if the ray is camera ray
    if(aux.type == RayType::CAMERA_RAY) return;

    // Check Material Sample Strategy
    assert(maxOutRay == 0);

    const bool isPathRayAsMISRay = renderState.directLightMIS && (aux.type == RayType::PATH_RAY);
    //const bool isCameraRay = aux.type == RayType::CAMERA_RAY;
    const bool isSpecularPathRay = aux.type == RayType::SPECULAR_PATH_RAY;
    // Always eval boundary mat if NEE is off
    // or NEE is on and hit endpoint and requested endpoint is same
    const GPULightI* requestedLight = renderState.gLightList[aux.endpointIndex];
    const bool isCorrectLight = (requestedLight->EndpointId() == gLight.EndpointId());
    const bool isCorrectNEERay = ((!renderState.nee) ||
                                  (isCorrectLight && aux.type == RayType::NEE_RAY));

    // If a path ray is hit
    bool isFirstDepthPathRay = (aux.depth == 1) && (aux.type == RayType::PATH_RAY);

    float misWeight = 1.0f;
    if(isPathRayAsMISRay)
    {
        Vector3 position = ray.ray.AdvancedPos(ray.tMax);
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
        // We are subsampling (discretely sampling) a single light
        // pdf of BxDF should also incorporate this
        float bxdfPDF = aux.prevPDF;
        misWeight = TracerFunctions::PowerHeuristic(1, bxdfPDF,
                                                    1, pdfLightC * pdfLightM);
    }

    // Accumulate Light if
    if(isCorrectNEERay      || // We hit the correct light as a NEE ray
       isPathRayAsMISRay    || // We hit a light with a path ray while MIS option is enabled
       isFirstDepthPathRay  || // These rays are "camera rays" of this surface
                               // which should not be culled when NEE is on
       isSpecularPathRay)      // We hit as spec ray which did not launched any NEE rays thus it should be hit
    {
        // Data Fetch
        const RayF& r = ray.ray;
        Vector3 position = r.AdvancedPos(ray.tMax);
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
        ImageAccumulatePixel(renderState.gImage, aux.pixelIndex, luminance);
    }
}

template <class MGroup>
__device__ __forceinline__
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
                       RandomGPU& rng,
                       // Constants
                       const typename MGroup::Data& gMatData,
                       const HitKey::Type matIndex)
{
        static constexpr Vector3 ZERO_3 = Zero3;

    // TODO: change this currently only first strategy is sampled
    static constexpr int PATH_RAY_INDEX = 0;
    static constexpr int NEE_RAY_INDEX  = 1;
    // Helper Class for better code readibiliy
    OutputWriter<RayAuxPath> outputWriter(gOutBoundKeys,
                                          gOutRays,
                                          gOutRayAux,
                                          maxOutRay);

    // Inputs
    // Current Ray
    const RayF& r = ray.ray;
    // Hit Position
    Vector3 position = r.AdvancedPos(ray.tMax);
    // Wi (direction is swapped as if it is coming out of the surface)
    Vector3 wi = -(r.getDirection().Normalize());
    // Current ray's medium
    const GPUMediumI& m = *(renderState.mediumList[aux.mediumIndex]);

    // Check Material Sample Strategy
    uint32_t sampleCount = maxOutRay;
    // Check Material's specularity;
    float specularity = MGroup::Specularity(surface, gMatData, matIndex);
    bool isSpecularMat = (specularity >= TracerConstants::SPECULAR_TRESHOLD);

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
            ImageAccumulatePixel(renderState.gImage,
                                 aux.pixelIndex,
                                 luminance);
        }
    }

    // If this material does not require to have any samples just quit
    if(sampleCount == 0) return;

    bool shouldLaunchMISRay = false;
    // ===================================== //
    //              NEE PORTION              //
    // ===================================== //
    // Dont launch NEE if not requested
    // or material is highly specula
    if(renderState.nee && !isSpecularMat &&
       (aux.type != RayType::CAMERA_RAY))
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
                              (aux.type != RayType::CAMERA_RAY));

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
            rayNEE.AdvanceSelf(MathConstants::Epsilon);
            RayReg rayOut;
            rayOut.ray = rayNEE;
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
        rayOut.tMax = INFINITY;
        // Aux
        RayAuxPath auxOut = aux;

        if(aux.type == RayType::CAMERA_RAY)
        {
            auxOut.pixelIndex = CalculateSphericalPixelId(rayPath.getDirection(),
                                                          renderState.resolution);
            ImageAddSample(renderState.gImage, auxOut.pixelIndex, 1);
        }

        auxOut.mediumIndex = static_cast<uint16_t>(outM->GlobalIndex());
        auxOut.radianceFactor = pathRadianceFactor;
        auxOut.type = (isSpecularMat) ? RayType::SPECULAR_PATH_RAY : RayType::PATH_RAY;
        // Save BxDF pdf if we hit a light
        // When we hit a light we will
        auxOut.prevPDF = pdfPath;
        auxOut.depth++;
        // Wrie
        outputWriter.Write(PATH_RAY_INDEX, rayOut, auxOut);
    }
    // All Done!
}