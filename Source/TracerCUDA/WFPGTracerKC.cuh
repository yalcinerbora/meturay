#pragma once

#include "RayAuxStruct.cuh"

#include "GPULightI.h"
#include "GPUMediumI.h"
#include "GPUDirectLightSamplerI.h"
#include "PathNode.cuh"
#include "RayStructs.h"
#include "ImageStructs.h"
#include "WorkOutputWriter.cuh"
#include "WFPGCommon.h"
#include "RaceSketch.cuh"

#include "TracerFunctions.cuh"
#include "TracerConstants.h"

#include "RayLib/RandomColor.h"

#include "AnisoSVO.cuh"
#include "GPUBlockPWCDistribution.cuh"
#include "RaceSketch.cuh"

static constexpr uint32_t INVALID_BIN_ID = std::numeric_limits<uint32_t>::max();

struct WFPGTracerGlobalState
{
    // Output Image
    ImageGMem<Vector4>              gImage;
    // Light Related
    const GPULightI**               gLightList;
    uint32_t                        totalLightCount;
    const GPUDirectLightSamplerI*   gLightSampler;
    // Medium Related
    const GPUMediumI**              mediumList;
    uint32_t                        totalMediumCount;
    // Path Guiding Related
    AnisoSVOctreeGPU                svo;
    RaceSketchGPU                   sketchGPU;
    // Path Related
    PathGuidingNode*                gPathNodes;
    uint32_t                        maximumPathNodePerRay;
    // Options
    // Path Guiding
    bool                            rawPathGuiding;
    // Options for NEE
    bool                            directLightMIS;
    bool                            nee;
    // Russian Roulette
    int                             rrStart;
};

struct WFPGTracerLocalState
{
    bool    emptyPrimitive;
};

__device__ inline
uint8_t DeterminePathIndexWFPG(uint8_t depth)
{
    return depth;
}

template <class EGroup>
__device__ inline
void WFPGTracerBoundaryWork(// Output
                            HitKey* gOutBoundKeys,
                            RayGMem* gOutRays,
                            RayAuxWFPG* gOutRayAux,
                            const uint32_t maxOutRay,
                            // Input as registers
                            const RayReg& ray,
                            const RayAuxWFPG& aux,
                            const typename EGroup::Surface& surface,
                            const RayId rayId,
                            // I-O
                            WFPGTracerLocalState& localState,
                            WFPGTracerGlobalState& renderState,
                            RNGeneratorGPUI& rng,
                            // Constants
                            const typename EGroup::GPUType& gLight)
{
    using GPUType = typename EGroup::GPUType;

    // Current Path
    const uint32_t pathStartIndex = aux.pathIndex * renderState.maximumPathNodePerRay;
    PathGuidingNode* gLocalPathNodes = renderState.gPathNodes + pathStartIndex;

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

    // Calculate the total contribution
    const RayF& r = ray.ray;
    Vector3 position = surface.WorldPosition();
    const GPUMediumI& m = *(renderState.mediumList[aux.mediumIndex]);

    // Calculate Transmittance factor of the medium
    Vector3 transFactor = m.Transmittance(ray.tMax);
    Vector3 radianceFactor = aux.radianceFactor * transFactor;

    Vector3 emission = gLight.Emit(// Input
                                   -r.getDirection(),
                                   position,
                                   surface);

    // And accumulate pixel// and add as a sample
    Vector3f total = emission * radianceFactor;
    // Incorporate MIS weight if applicable
    // if path ray hits a light misWeight is calculated
    // else misWeight is 1.0f
    total *= misWeight;

    // Previous Path's index
    int8_t prevDepth = aux.depth - 1;
    uint8_t prevPathIndex = DeterminePathIndexWFPG(prevDepth);

    // Accumulate the contribution if
    if(isPathRayNEEOff   || // We hit a light with a path ray while NEE is off
       isPathRayAsMISRay || // We hit a light with a path ray while MIS option is enabled
       isCorrectNEERay   || // We hit the correct light as a NEE ray while NEE is on
       isCameraRay       || // We hit as a camera ray which should not be culled when NEE is on
       isSpecularPathRay)   // We hit as spec ray which did not launched any NEE rays thus it should contribute
    {
        // Accumulate the pixel
        ImageAccumulatePixel(renderState.gImage,
                             aux.pixelIndex,
                             Vector4f(total, 1.0f));

        // Also back propagate this radiance to the path nodes
        if(aux.type != RayType::CAMERA_RAY)
           // &&
           //// If current path is the first vertex in the chain skip
           //pathIndex != 0)
        {
            // Generate Spherical Coordinates
            Vector3f wi = ray.ray.getDirection().Normalize();
            Vector3 dirZUp = Vector3(wi[2], wi[0], wi[1]);
            Vector2f sphrCoords = Utility::CartesianToSphericalUnit(dirZUp);
            // Add to the hash table
            //renderState.sketchGPU.AtomicAddData(r.getPosition(), sphrCoords,
            //                                    Utility::RGBToLuminance(emission));

            // Accumulate Radiance from 2nd vertex (including this vertex) away
            // S-> surface
            // C-> camera
            // L-> light
            //
            // C --- S --- S --- S --- L
            //                   |     ^
            //                   |     We are here (pathIndex points here)
            //                   v
            //                  we should accum-down from here
            gLocalPathNodes[prevPathIndex - 1].AccumRadianceDownChain(total, gLocalPathNodes);
        }
    }
}

template <class MGroup>
__device__ inline
void WFPGTracerPathWork(// Output
                        HitKey* gOutBoundKeys,
                        RayGMem* gOutRays,
                        RayAuxWFPG* gOutRayAux,
                        const uint32_t maxOutRay,
                        // Input as registers
                        const RayReg& ray,
                        const RayAuxWFPG& aux,
                        const typename MGroup::Surface& surface,
                        const RayId rayId,
                        // I-O
                        WFPGTracerLocalState& localState,
                        WFPGTracerGlobalState& renderState,
                        RNGeneratorGPUI& rng,
                        // Constants
                        const typename MGroup::Data& gMatData,
                        const HitKey::Type matIndex)
{
    static constexpr Vector3 ZERO_3 = Zero3;

    // Path Memory
    const uint32_t pathStartIndex = aux.pathIndex * renderState.maximumPathNodePerRay;
    PathGuidingNode* gLocalPathNodes = renderState.gPathNodes + pathStartIndex;

    // TODO: change this currently only first strategy is sampled
    static constexpr int PATH_RAY_INDEX = 0;
    static constexpr int NEE_RAY_INDEX  = 1;
    // Helper Class for better code readability
    OutputWriter<RayAuxWFPG> outputWriter(gOutBoundKeys,
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
        ImageAccumulatePixel(renderState.gImage,
                             aux.pixelIndex,
                             Vector4f(total, 1.0f));

        // Accumulate this to the paths as well
        if(total.HasNaN()) printf("NAN Found emissive!!!\n");
        gLocalPathNodes[aux.depth].AccumRadianceDownChain(total, gLocalPathNodes);
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
                              // dirac delta light (point light spot light etc.)
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
            rayNEE.NudgeSelf(surface.WorldGeoNormal(), surface.curvatureOffset);
            RayReg rayOut;
            rayOut.ray = rayNEE;
            rayOut.tMin = 0.0f;
            rayOut.tMax = lDistance;
            // Aux
            RayAuxWFPG auxOut = aux;
            auxOut.radianceFactor = neeRadianceFactor;
            auxOut.endpointIndex = lightIndex;
            auxOut.type = RayType::NEE_RAY;
            auxOut.prevPDF = NAN;
            auxOut.depth++;
            outputWriter.Write(NEE_RAY_INDEX, rayOut, auxOut, matLight);
        }
    }

    // ==================================== //
    //             BxDF PORTION             //
    // ==================================== //
    // Sample a path from material
    RayF rayPath; float pdfPath; const GPUMediumI* outM = &m;
    Vector3f reflectance;
    // Sample a path using SDTree
    if(!isSpecularMat)
    {
        constexpr float BxDF_GuideSampleRatio = 0.0f;
        float xi = rng.Uniform();

        bool selectedPDFZero = false;
        float pdfBxDF, pdfGuide;
        if(xi >= BxDF_GuideSampleRatio)
        {
            // Sample using BxDF
            reflectance = MGroup::Sample(// Outputs
                                         rayPath, pdfBxDF, outM,
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
            pdfGuide = aux.guidePDF;

            if(pdfBxDF == 0.0f) selectedPDFZero = true;
        }
        else
        {
            // Sample a path using SDTree
            Vector3f direction = Utility::SphericalToCartesianUnit(Vector2f(aux.guideDir[0],
                                                                            aux.guideDir[1]));
            pdfGuide = aux.guidePDF;

            // Calculate BxDF
            reflectance = MGroup::Evaluate(// Input
                                           direction,
                                           wi,
                                           position,
                                           m,
                                           //
                                           surface,
                                           // Constants
                                           gMatData,
                                           matIndex);

            pdfBxDF = MGroup::Pdf(direction,
                                  wi,
                                  position,
                                  m,
                                  //
                                  surface,
                                  gMatData,
                                  matIndex);

            // Generate a ray using the values
            rayPath = RayF(direction, position);
            rayPath.NudgeSelf(surface.WorldGeoNormal(), surface.curvatureOffset);

            if(pdfGuide == 0.0f) selectedPDFZero = true;
        }
        // Pdf Average
        //pdfPath = pdfBxDF;
        pdfPath = BxDF_GuideSampleRatio          * pdfGuide +
                  (1.0f - BxDF_GuideSampleRatio) * pdfBxDF;
        pdfPath = selectedPDFZero ? 0.0f : pdfPath;

        // DEBUG
        if(isnan(pdfPath) || isnan(pdfBxDF) || isnan(pdfGuide))
            printf("[%s] NAN PDF = % f = w * %f + (1.0f - w) * %f, w: % f\n",
                   (xi >= BxDF_GuideSampleRatio) ? "BxDF": "Tree",
                   pdfPath, pdfBxDF, pdfGuide, BxDF_GuideSampleRatio);
        if(pdfPath != 0.0f && rayPath.getDirection().HasNaN())
            printf("[%s] NAN DIR %f, %f, %f\n",
                    (xi >= BxDF_GuideSampleRatio) ? "BxDF" : "Tree",
                    rayPath.getDirection()[0],
                    rayPath.getDirection()[1],
                    rayPath.getDirection()[2]);
        if(reflectance.HasNaN())
            printf("[%s] NAN REFL %f %f %f\n",
                   (xi >= BxDF_GuideSampleRatio) ? "BxDF" : "Tree",
                   reflectance[0],
                   reflectance[1],
                   reflectance[2]);

        //if(isnan(pdfPath) || isnan(pdfBxDF) || isnan(pdfGuide) ||
        //   rayPath.getDirection().HasNaN() || reflectance.HasNaN())
        //    return;
    }
    else
    {
        // Specular Mat, Only sample using BxDF
        reflectance = MGroup::Sample(// Outputs
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

    }

     // Factor the radiance of the surface
    Vector3f pathRadianceFactor = radianceFactor * reflectance;
    // Check singularities
    pathRadianceFactor = (pdfPath == 0.0f) ? Zero3 : (pathRadianceFactor / pdfPath);

    if(pathRadianceFactor.HasNaN())
        printf("NAN PATH R: %f %f %f = {%f %f %f} * {%f %f %f} / %f  \n",
               pathRadianceFactor[0], pathRadianceFactor[1], pathRadianceFactor[2],
               radianceFactor[0], radianceFactor[1], radianceFactor[2],
               reflectance[0], reflectance[1], reflectance[2], pdfPath);

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
        RayAuxWFPG auxOut = aux;
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
    // Record this intersection on path chain
    uint8_t prevPathIndex = DeterminePathIndexWFPG(aux.depth - 1);
    uint8_t curPathIndex = DeterminePathIndexWFPG(aux.depth);

    PathGuidingNode node;
    //printf("WritingNode PC:(%u %u) W:(%f, %f, %f) RF:(%f, %f, %f) Path: %u\n",
    //       static_cast<uint32_t>(prevPathIndex), static_cast<uint32_t>(curPathIndex),
    //       position[0], position[1], position[2],
    //       pathRadianceFactor[0], pathRadianceFactor[1], pathRadianceFactor[2],
    //       aux.pathIndex);
    node.prevNext[1] = PathGuidingNode::InvalidIndex;
    node.prevNext[0] = prevPathIndex;
    node.worldPosition = position;
    // Unlike other techniques that holds incoming radiance
    // WFPG holds outgoing radiance. To calculate that, wee need to store
    // previous paths throughput
    //node.radFactor = aux.radianceFactor;
    node.radFactor = pathRadianceFactor;

    node.totalRadiance = Zero3;
    gLocalPathNodes[curPathIndex] = node;
    // Set Previous Path node's next index
    if(prevPathIndex != PathGuidingNode::InvalidIndex)
        gLocalPathNodes[prevPathIndex].prevNext[1] = curPathIndex;

    // All Done!
}

template <class MGroup>
__device__ inline
void WFPGTracerPhotonWork(// Output
                          HitKey* gOutBoundKeys,
                          RayGMem* gOutRays,
                          RayAuxPhotonWFPG* gOutRayAux,
                          const uint32_t maxOutRay,
                          // Input as registers
                          const RayReg& ray,
                          const RayAuxPhotonWFPG& aux,
                          const typename MGroup::Surface& surface,
                          const RayId rayId,
                          // I-O
                          WFPGTracerLocalState& localState,
                          WFPGTracerGlobalState& renderState,
                          RNGeneratorGPUI& rng,
                          // Constants
                          const typename MGroup::Data& gMatData,
                          const HitKey::Type matIndex)
{
    assert(maxOutRay == 1);

    static constexpr Vector3 ZERO_3 = Zero3;
    // TODO: change this currently only first strategy is sampled
    static constexpr int PHOTON_RAY_INDEX = 0;
    // Helper Class for better code readability
    OutputWriter<RayAuxPhotonWFPG> outputWriter(gOutBoundKeys,
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

    // Calculate Transmittance factor of the medium
    // And reduce the radiance wrt the medium transmittance
    Vector3 transFactor = m.Transmittance(ray.tMax);
    Vector3 photonReceivedPower = aux.power * transFactor;

    // Check Material's specularity;
    float specularity = MGroup::Specularity(surface, gMatData, matIndex);
    bool isSpecularMat = (specularity >= TracerConstants::SPECULAR_TRESHOLD);

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
    Vector3f photonScatteredFactor = photonReceivedPower * reflectance;
    // Check singularities
    photonScatteredFactor = (pdfPath == 0.0f) ? Zero3 : (photonScatteredFactor / pdfPath);

    // Check Russian Roulette
    float avgThroughput = photonScatteredFactor.Dot(Vector3f(0.333f));
    bool terminateRay = ((aux.depth > renderState.rrStart) &&
                         TracerFunctions::RussianRoulette(photonScatteredFactor, avgThroughput, rng));


    if(photonScatteredFactor.HasNaN() || photonReceivedPower.HasNaN() ||
       isnan(pdfPath))
        printf("NAN PHOTON R: %f %f %f = {%f %f %f} * {%f %f %f} / %f  \n",
               photonScatteredFactor[0], photonScatteredFactor[1], photonScatteredFactor[2],
               photonReceivedPower[0], photonReceivedPower[1], photonReceivedPower[2],
               reflectance[0], reflectance[1], reflectance[2], pdfPath);

    // Do not terminate rays ever for specular mats
    if((!terminateRay || isSpecularMat) &&
       // Do not waste rays on zero radiance paths
       photonScatteredFactor != ZERO_3)
    {
        if(!isSpecularMat &&  // Only non-specular surfaces can hold photons
           (aux.depth == 1))  // Don't store direct illumination photons
                              // (we are concerned with indirect illumination guide)
        {
            // Store the photon power to the sketch
            // "incoming direction towards this position"
            // Generate Spherical Coordinates
            Vector3f wi = ray.ray.getDirection().Normalize();
            Vector3 dirZUp = Vector3(wi[2], wi[0], wi[1]);
            Vector2f sphrCoords = Utility::CartesianToSphericalUnit(dirZUp);
            // Store operation
            float luminance = Utility::RGBToLuminance(photonReceivedPower);

            // TEST Firefly elimination
            if(luminance < 30.0f)
                renderState.sketchGPU.AtomicAddData(position,
                                                    sphrCoords,
                                                    luminance);

        };

        // Ray
        RayReg rayOut;
        rayOut.ray = rayPath;
        rayOut.tMin = 0.0f;
        rayOut.tMax = FLT_MAX;
        // Aux
        RayAuxPhotonWFPG auxOut;
        auxOut.mediumIndex = static_cast<uint16_t>(outM->GlobalIndex());
        auxOut.power = photonScatteredFactor;
        auxOut.depth = aux.depth + 1;
        // Write
        outputWriter.Write(PHOTON_RAY_INDEX, rayOut, auxOut);
    }
}

template <class EGroup>
__device__ inline
void WFPGTracerDebugBWork(// Output
                          HitKey* gOutBoundKeys,
                          RayGMem* gOutRays,
                          RayAuxWFPG* gOutRayAux,
                          const uint32_t maxOutRay,
                          // Input as registers
                          const RayReg& ray,
                          const RayAuxWFPG& aux,
                          const typename EGroup::Surface& surface,
                          const RayId rayId,
                          // I-O
                          WFPGTracerLocalState& localState,
                          WFPGTracerGlobalState& renderState,
                          RNGeneratorGPUI& rng,
                          // Constants
                          const typename EGroup::GPUType& gLight)
{
    const AnisoSVOctreeGPU& svo = renderState.svo;
    // Helper Class for better code readability
    OutputWriter<RayAuxWFPG> outputWriter(gOutBoundKeys,
                                          gOutRays,
                                          gOutRayAux,
                                          maxOutRay);

    // Only Direct Hits from camera are used
    // In debugging
    if(aux.depth != 1) return;
    // Hit Position
    Vector3 position = surface.WorldPosition();

    // Query SVO find a leaf
    uint32_t svoLeafIndex;
    bool found = svo.LeafIndex(svoLeafIndex, position, true);
    Vector3f locColor = (found) ? Utility::RandomColorRGB(svoLeafIndex)
                                : Vector3f(0.0f);
    // Accumulate the pixel
    ImageAccumulatePixel(renderState.gImage,
                            aux.pixelIndex,
                            Vector4f(locColor, 1.0f));
}

template <class MGroup>
__device__ inline
void WFPGTracerDebugWork(// Output
                         HitKey* gOutBoundKeys,
                         RayGMem* gOutRays,
                         RayAuxWFPG* gOutRayAux,
                         const uint32_t maxOutRay,
                         // Input as registers
                         const RayReg& ray,
                         const RayAuxWFPG& aux,
                         const typename MGroup::Surface& surface,
                         const RayId rayId,
                         // I-O
                         WFPGTracerLocalState& localState,
                         WFPGTracerGlobalState& renderState,
                         RNGeneratorGPUI& rng,
                         // Constants
                         const typename MGroup::Data& gMatData,
                         const HitKey::Type matIndex)
{
    const AnisoSVOctreeGPU& svo = renderState.svo;
    // Helper Class for better code readability
    OutputWriter<RayAuxWFPG> outputWriter(gOutBoundKeys,
                                          gOutRays,
                                          gOutRayAux,
                                          maxOutRay);
    // Only Direct Hits from camera are used
    // In debugging
    if(aux.depth != 1) return;
    // Hit Position
    Vector3 position = surface.WorldPosition();

    // Query SVO find a leaf
    uint32_t svoLeafIndex;
    bool found = svo.LeafIndex(svoLeafIndex, position, true);
    Vector3f locColor = (found) ? Utility::RandomColorRGB(svoLeafIndex)
                                : Vector3f(0.0f);
    // Accumulate the pixel
    ImageAccumulatePixel(renderState.gImage,
                            aux.pixelIndex,
                            Vector4f(locColor, 1.0f));
}

__global__ CUDA_LAUNCH_BOUNDS_1D
static void KCTraceSVO(// Output
                       ImageGMem<Vector4> gImage,
                       // Input
                       const AnisoSVOctreeGPU svo,
                       const RayGMem* gRays,
                       const RayAuxWFPG* gRayAux,
                       // Constants
                       WFPGRenderMode mode,
                       uint32_t rayCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < rayCount;
        threadId += (blockDim.x * gridDim.x))
    {
        RayReg ray = RayReg(gRays, threadId);
        RayAuxWFPG aux = gRayAux[threadId];

        uint32_t svoLeafIndex;
        float tMin = svo.TraceRay(svoLeafIndex, ray.ray,
                                  ray.tMin, ray.tMax);

        Vector3f locColor = Vector3f(1.0f, 0.0f, 1.0f);
        if(mode == WFPGRenderMode::SVO_FALSE_COLOR)
            locColor = (svoLeafIndex != UINT32_MAX) ? Utility::RandomColorRGB(svoLeafIndex)
                                                    : Vector3f(0.0f);
        else if(mode == WFPGRenderMode::SVO_RADIANCE)
        {
            half radiance = svo.ReadRadiance(svoLeafIndex, true,
                                             -ray.ray.getDirection());
            float radianceF = radiance;
            if(svoLeafIndex != UINT32_MAX)
                locColor = Vector3f(radianceF);
            else
                locColor = Vector3f(1.0f, 0.0f, 1.0f);
        }

        // Accumulate the pixel
        ImageAccumulatePixel(gImage,
                             aux.pixelIndex,
                             Vector4f(locColor, 1.0f));
    }
}

__global__ CUDA_LAUNCH_BOUNDS_1D
static void KCQuerySketch(// Output
                          ImageGMem<Vector4> gImage,
                          // Input
                          const AnisoSVOctreeGPU svo,
                          const RaceSketchGPU sketch,
                          const RayGMem* gRays,
                          const RayAuxWFPG* gRayAux,
                          // Constants
                          uint32_t rayCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < rayCount;
        threadId += (blockDim.x * gridDim.x))
    {
        RayReg ray = RayReg(gRays, threadId);
        RayAuxWFPG aux = gRayAux[threadId];

        //uint32_t svoLeafIndex;
        //float tMin = svo.TraceRay(svoLeafIndex, ray.ray,
        //                          ray.tMin, ray.tMax);

        //Vector3f pos = ray.ray.AdvancedPos(tMin);
        Vector3f pos = ray.ray.getPosition();

        // Generate Spherical Coordinates
        Vector3f wo = ray.ray.getDirection();
        Vector3 dirZUp = Vector3(wo[2], wo[0], wo[1]);
        Vector2f sphrCoords = Utility::CartesianToSphericalUnit(dirZUp);

        float prob = sketch.Probability(pos, sphrCoords);
        Vector3f locColor = Vector3f(prob);
        // Accumulate the pixel
        ImageAccumulatePixel(gImage,
                             aux.pixelIndex,
                             Vector4f(locColor, 1.0f));
    }
}

__device__ inline
uint32_t GenerateLeafIndex(uint32_t leafIndex)
{
    static constexpr uint32_t LAST_BIT_UINT32 = 31;
    uint32_t result = leafIndex;
    result |= (1u << LAST_BIT_UINT32);
    return result;
}

__device__ inline
uint32_t GenerateNodeIndex(uint32_t nodeIndex)
{
    return nodeIndex;
}

__device__ inline
bool ReadSVONodeId(uint32_t& nodeId, uint32_t packedData)
{
    static constexpr uint32_t LAST_BIT_UINT32 = 31;
    static constexpr uint32_t INDEX_MASK = (1u << LAST_BIT_UINT32) - 1;

    bool isLeaf = (packedData >> LAST_BIT_UINT32) & 0b1;
    nodeId = packedData & INDEX_MASK;
    return isLeaf;
}

__device__ inline
Vector3f DirIdToWorldDir(const Vector2ui& dirXY,
                         const Vector2ui& dimensions)
{
    assert(dirXY < dimensions);
    using namespace MathConstants;
    // Spherical coordinate deltas
    Vector2f deltaXY = Vector2f((2.0f * Pi) / static_cast<float>(dimensions[0]),
                                Pi / static_cast<float>(dimensions[1]));

    // Assume image space bottom left is (0,0)
    // Center to the pixel as well
    Vector2f dirXYFloat = Vector2f(dirXY[0], dirXY[1]) + Vector2f(0.5f);
    Vector2f sphrCoords = Vector2f(dirXYFloat[0] * deltaXY[0],
                                   Pi - dirXYFloat[1] * deltaXY[0]);

    return Utility::SphericalToCartesianUnit(sphrCoords);
}

__global__
static void KCInitializeSVOBins(// Outputs
                                RayAuxWFPG* gRayAux,
                                // Inputs
                                const RayGMem* gRays,
                                const HitKey* gRayWorkKeys,
                                // Constants
                                HitKey boundaryMatKey,
                                AnisoSVOctreeGPU svo,
                                uint32_t rayCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < rayCount;
        threadId += (blockDim.x * gridDim.x))
    {
        const RayType rayType = gRayAux[threadId].type;
        RayReg ray = RayReg(gRays, threadId);
        HitKey key = gRayWorkKeys[threadId];

        bool unnecessaryRay = (rayType == RayType::NEE_RAY ||
                               key == HitKey::InvalidKey ||
                               key == boundaryMatKey);

        // Skip NEE rays
        // These either hit the light or not
        // For both cases it is better to skip them
        // Or if the ray is invalid
        // It happens when a material does not write a ray
        // to its available spot)
        if(unnecessaryRay)
        {
            gRayAux[threadId].binId = INVALID_BIN_ID;
            continue;
        }

        // Hit Position
        Vector3 position = ray.ray.AdvancedPos(ray.tMax);

        // Find the leaf
        // Traverse all 8 neighbors of the ray hit position
        // due to numerical inaccuracies
        uint32_t leafIndex;
        bool found = svo.LeafIndex(leafIndex, position, true);

        // DEBUG
        if(!found) printf("Leaf Not found!\n");

        // Store the found leaf & increment the ray count
        // for that leaf
        if(found)
        {
            gRayAux[threadId].binId = GenerateLeafIndex(leafIndex);
            // Increment the ray count on that leaf
            svo.IncrementLeafRayCount(leafIndex);
        }
    }
}

__global__
static void KCCheckReducedSVOBins(// I-O
                                  RayAuxWFPG* gRayAux,
                                  // Constants
                                  AnisoSVOctreeGPU svo,
                                  uint32_t rayCount)
{
    for(uint32_t threadId = threadIdx.x + blockDim.x * blockIdx.x;
        threadId < rayCount;
        threadId += (blockDim.x * gridDim.x))
    {
        // Skip if this ray is not used
        if(gRayAux[threadId].binId == INVALID_BIN_ID)
            continue;

        uint32_t leafNodeId;
        ReadSVONodeId(leafNodeId, gRayAux[threadId].binId);

        bool isLeaf;
        uint32_t newBinId = svo.FindMarkedBin(isLeaf, leafNodeId);

        if(!isLeaf)
        {
            gRayAux[threadId].binId = newBinId;
        }
    }
}

// Main directional distribution generation kernel
// Each block is responsible of a single bin
// Each block will trace the SVO to generate
//  omni-directional distribution map.
template <class RNG, uint32_t THREAD_PER_BLOCK, uint32_t X, uint32_t Y>
__global__ __launch_bounds__(THREAD_PER_BLOCK)
static void KCGenAndSampleDistribution(// Output
                                       RayAuxWFPG* gRayAux,
                                       // I-O
                                       RNGeneratorGPUI** gRNGs,
                                       // Input
                                       // Per-ray
                                       const RayGMem* gRays,
                                       const RayId* gRayIds,
                                       // Per bin
                                       const uint32_t* gBinOffsets,
                                       const uint32_t* gNodeIds,
                                       // Constants
                                       const AnisoSVOctreeGPU svo,
                                       uint32_t binCount)
{
    auto& rng = RNGAccessor::Acquire<RNG>(gRNGs, LINEAR_GLOBAL_ID);
    auto IsMainThread = []() -> bool
    {
        return threadIdx.x == 0;
    };
    const uint32_t THREAD_ID = threadIdx.x;

    // Number of threads that contributes to the ray tracing operation
    static constexpr uint32_t RT_CONTRIBUTING_THREAD_COUNT = (THREAD_PER_BLOCK < X * Y) ? THREAD_PER_BLOCK : (X * Y);
    // How many rows can we process in parallel
    static constexpr uint32_t ROW_PER_ITERATION = RT_CONTRIBUTING_THREAD_COUNT / X;
    // How many iterations the entire image would take
    static constexpr uint32_t ROW_ITER_COUNT = Y / ROW_PER_ITERATION;
    static_assert(RT_CONTRIBUTING_THREAD_COUNT % X == 0, "RT_THREADS must be multiple of X or vice versa.");
    static_assert(Y % ROW_PER_ITERATION == 0, "RT_THREADS must exactly iterate over X * Y");
    // Ray tracing related
    static constexpr uint32_t RT_ITER_COUNT = ROW_ITER_COUNT;
    // PWC Distribution over the shared memory
    using BlockPWC2D = BlockPWCDistribution2D<THREAD_PER_BLOCK, X, Y>;

    // Allocate shared memory for PWC Distribution
    __shared__ typename BlockPWC2D::TempStorage sPWCMem;
    // Starting positions of the rays (at most TPB)
    __shared__ Vector3f sPositions[THREAD_PER_BLOCK];
    // Bin parameters
    __shared__ uint32_t sRayCount;
    __shared__ uint32_t sOffsetStart;
    __shared__ uint32_t sNodeId;
    __shared__ uint32_t sPositionCount;
    __shared__ uint32_t sBinVoxelSize;

    // For each block (we allocate enough blocks for the GPU)
    // Each block will process multiple bins
    for(uint32_t binIndex = blockIdx.x; binIndex < binCount;
        binIndex += gridDim.x)
    {
        // Load Bin information
        if(IsMainThread())
        {
            Vector2ui rayRange = Vector2ui(gBinOffsets[binIndex], gBinOffsets[binIndex + 1]);
            sRayCount = rayRange[1] - rayRange[0];
            sOffsetStart = rayRange[0];
            sNodeId = gNodeIds[binIndex];
            sPositionCount = min(THREAD_PER_BLOCK, sRayCount);

            // Calculate the voxel size of the bin
            uint32_t nodeId;
            bool isLeaf = ReadSVONodeId(nodeId, sNodeId);
            sBinVoxelSize = svo.NodeVoxelSize(nodeId, isLeaf);
        }
        __syncthreads();

        // Kill the entire block if Node Id is invalid
        if(sNodeId == INVALID_BIN_ID) continue;

        // Load positions of the rays to the shared memory
        // Try to load minimum of TPB or ray count
        if(THREAD_ID < sPositionCount)
        {
            uint32_t rayId = gRayIds[sOffsetStart + THREAD_ID];
            RayReg ray(gRays, rayId);
            // Hit Position
            Vector3 position = ray.ray.AdvancedPos(ray.tMax);
            sPositions[THREAD_ID] = position;
        }
        __syncthreads();

        // Incoming radiance (result of the trace)
        float incRadiances[RT_ITER_COUNT];

        // Trace the rays
        for(uint32_t i = 0; i < RT_ITER_COUNT; i++)
        {
            // This case occurs only when there is more threads than pixels
            if(THREAD_ID >= RT_CONTRIBUTING_THREAD_COUNT) continue;

            // Determine your direction
            uint32_t directionId = (i * THREAD_PER_BLOCK) + THREAD_ID;
            Vector2ui dirIdXY = Vector2ui(directionId % X,
                                          directionId / X);
            Vector3f worldDir = DirIdToWorldDir(dirIdXY, Vector2ui(X, Y));
            // Get a random position from the pool
            Vector3f position = sPositions[THREAD_ID % sPositionCount];

            // Now this is the interesting part
            // We need to offset the ray in order to prevent
            // self intersections.
            // However it is not easy since we use arbitrary locations
            // over the voxel, most fool-proof (but inaccurate)
            // way is to offset the ray with the current level voxel size.
            // This will be highly inaccurate when current bin(node) level is low.
            // TODO: Change it to a better solution
            float tMin = sBinVoxelSize * MathConstants::Sqrt3 + MathConstants::LargeEpsilon;

            uint32_t leafId;
            svo.TraceRay(leafId, RayF(worldDir, position),
                         tMin, FLT_MAX);
            incRadiances[i] = 1.0f;// svo.ReadRadiance(leafId, true, -worldDir);
        }
        // We finished tracing rays from the scene
        // Now generate distribution from the data
        // and sample for each ray

        // Generate PWC Distribution over the radiances
        BlockPWC2D dist2D(sPWCMem, incRadiances);
        // Block threads will loop over the every ray in this bin
        for(uint32_t rayIndex = THREAD_ID; rayIndex < sRayCount;
            rayIndex += THREAD_PER_BLOCK)
        {
            float pdf;
            Vector2f index;
            Vector2f uv = dist2D.Sample(pdf, index, rng);
            //Vector2f uv = Zero2f;
            //float pdf = 0.0f;

            // Store the sampled direction of the ray
            uint32_t rayId = gRayIds[sOffsetStart + rayIndex];
            gRayAux[rayId].guideDir = Vector2h(uv[0], uv[1]);
            gRayAux[rayId].guidePDF = pdf;
        }

        // Sync every thread before processing another bin
        __syncthreads();

        //for(uint32_t rayIndex = 0; rayIndex < sRayCount; rayIndex++)
        //{
        //    incRadiances[0] *= rayIndex * 10.f;
        //    incRadiances[1] *= rayIndex * 10.f;
        //    BlockPWC2D dist2D(sPWCMem, incRadiances);

        //    if(IsMainThread())
        //    {
        //        float pdf;
        //        Vector2f index;
        //        Vector2f uv = dist2D.Sample(pdf, index, rng);

        //        uint32_t rayId = gRayIds[sOffsetStart + rayIndex];
        //        // Store the sampled direction of the ray
        //        gRayAux[rayId].guideDir = Vector2h(uv[0], uv[1]);
        //        gRayAux[rayId].guidePDF = pdf;
        //    }
        //    __syncthreads();
        //}
    }
}